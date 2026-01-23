# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.layers.wrappers import move_device_like
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.mask_head import BaseMaskRCNNHead
from detectron2.utils.registry import Registry
from ..selsa.aggerator import SelsaAggregator

__all__ = [   
    "MaskRCNNConvUpsampleHead2",
    "build_mask_head",
]


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def focal_loss(pred_mask_logits, gt_masks, alpha=0.25, gamma=2.0):
    #pred_prob = torch.sigmoid(pred_mask_logits)
    pred_prob = pred_mask_logits

    bce_loss = F.binary_cross_entropy(pred_prob, gt_masks, reduction='mean')
    alpha_factor = torch.where(gt_masks == 1, alpha, 1 - alpha)
    focal_weight = torch.where(gt_masks == 1, 1 - pred_prob, pred_prob)
    focal_weight = alpha_factor * (focal_weight ** gamma)
    loss = focal_weight * bce_loss
    return loss.mean()






@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead2(BaseMaskRCNNHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        #self.conv_norm_relus =[]

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )            
            self.add_module(f"mask_fcn{k + 1}", conv)
            #self.conv_norm_relus[f"mask_fcn{k + 1}"] = conv
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv", self.deconv)
        #self.relu = nn.ReLU()
        self.add_module("deconv_relu",  nn.ReLU())
        cur_channels = conv_dims[-1]
        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.add_module("predictor", self.predictor)
        
        self.aggregator = SelsaAggregator(in_channels=784, num_attention_blocks=4)
        self.add_module("aggregator", self.aggregator)
        #self.fc = nn.Linear(256, 256)
        self.inplace_false_relu = nn.ReLU(inplace=False).cuda()
        self.add_module("inplace_false_relu", self.inplace_false_relu)
        
        for name, layer in self.named_children():
            if isinstance(layer, (Conv2d, ConvTranspose2d)):
                weight_init.c2_msra_fill(layer)
        #conv_norm_relus_list = list(self.conv_norm_relus)
        #layers = conv_norm_relus_list + [self.deconv]
        #for layer in layers:
        #    weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x):
        #for layer in self:
        #    x=layer(x)
        for name, module in self.named_children():
            if name not in ["deconv", "predictor", "deconv_relu", "aggregator", "inplace_false_relu"]:
                x = module(x)
        x = self.deconv(x)
        x = self.deconv_relu(x)
        x = self.predictor(x)
        return x

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        #print('mask:',x.size())
        x = self.layers(x)
        
        if self.training:
            #return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period) * self.loss_weight}
            return {"loss_mask": self.mask_rcnn_loss(x, instances, 1000) * self.loss_weight}
        else:
            
            self.mask_rcnn_inference(x, instances)
            return instances

    
    def mask_aggregator(self,pred_mask_logits, instances,training=True):

        #pred_masks =torch.sigmoid(pred_mask_logits)
        if training == True:
            number_per_instance = [instance.proposal_boxes.tensor.size(0) for instance in instances]
        else:
            number_per_instance = [instance.pred_boxes.tensor.size(0) for instance in instances]
            pred_mask_logits = torch.squeeze(pred_mask_logits)
            #print('pred_mask_logits.dim()',pred_mask_logits.dim())
            if pred_mask_logits.dim() <3:
                torch.unsqueeze(pred_mask_logits, 0) 
        start_idx = 0
        chunks = []
        #print(pred_mask_logits.size())
        # Slice the feature tensor based on proposal counts
        for size in number_per_instance:
            end_idx = start_idx + size
            chunk = pred_mask_logits[start_idx:end_idx, :, :]  # Slice along the first dimension
            chunks.append(chunk)
            start_idx = end_idx  # Update the start index for the next chunk

        aggregated_features = []
        if len(chunks)==1:
            aggregated_features=chunks
        else:
            for i in range(len(chunks)):
                now_feature = chunks[i]  # Current feature chunk
                
                ref_features = [chunks[j] for j in range(len(chunks)) if j != i]  # Reference chunks
                ref_features = torch.cat(ref_features, dim=0)  # Concatenate reference chunks
                #print('now_feature',now_feature.size())
                #print('ref_features',ref_features.size())
                fea_deltas_now = now_feature.view(now_feature.size(0), now_feature.size(1)*now_feature.size(2))
                fea_deltas_ref = ref_features.view(ref_features.size(0), ref_features.size(1)*ref_features.size(2))
                # Compute aggregated features
                #fea_deltas_now = self.avg_pool(now_feature).flatten(1)
                #fea_deltas_ref = self.avg_pool(ref_features).flatten(1)

                for stage in range(0,2):  # Iterate over the aggregation stages
                    x_aggre = self.aggregator(fea_deltas_now, fea_deltas_ref)
                    fea_deltas_now = fea_deltas_now + x_aggre
                    fea_deltas_now = self.inplace_false_relu(fea_deltas_now)
                fea_deltas_now= torch.reshape(fea_deltas_now,(fea_deltas_now.size(0),28,28))
                aggregated_features.append(fea_deltas_now)
            # Append aggregated feature for the current c
        pred_mask_logits = torch.cat(aggregated_features,dim=0)
        pred_mask_logits = torch.clamp(pred_mask_logits, min=0.0, max=1.0)
        
        #print('pre::',pred_mask_logits.size())
        if training ==False:
            pred_mask_logits = torch.logit(pred_mask_logits)
            pred_mask_logits = torch.unsqueeze(pred_mask_logits,1)
            #print(pred_mask_logits.size())
        return pred_mask_logits
    
    def mask_rcnn_loss(self,pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int=0):
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0) #110
        mask_side_len = pred_mask_logits.size(2) #28
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
        
        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)
            #print('gt_masks_per_image',gt_masks_per_image.size())
        
        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)
        
        if cls_agnostic_mask:
            #[106, 1, 28, 28]
            pred_mask_logits = pred_mask_logits[:, 0] #[106, 28, 28]
            
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        # Log the training accuracy (using gt classes and sigmoid(0.0) == 0.5 threshold)
        mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = gt_masks_bool.sum().item()
        false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
            gt_masks_bool.numel() - num_positive, 1.0
        )
        false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

        
        storage = get_event_storage()
        storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
        storage.put_scalar("mask_rcnn/false_positive", false_positive)
        #storage.put_scalar("mask_rcnn/false_negative", false_negative)
        if vis_period > 0 and storage.iter % vis_period == 0:
            pred_masks = pred_mask_logits.sigmoid()
            vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
            name = "Left: mask prediction;   Right: mask GT"
            for idx, vis_mask in enumerate(vis_masks):
                vis_mask = torch.stack([vis_mask] * 3, axis=0)
                storage.put_image(name + f" ({idx})", vis_mask)
        
        
        ##########################
        
        #pred_mask_logits = torch.sigmoid(pred_mask_logits)
        #pred_mask_logits = self.mask_aggregator(pred_mask_logits,instances)
       
        
        ########################
        
        #mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
        mask_loss = focal_loss(pred_mask_logits, gt_masks)
        return mask_loss


    def mask_rcnn_inference(self,pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
        """
        Convert pred_mask_logits to estimated foreground probability masks while also
        extracting only the masks for the predicted classes in pred_instances. For each
        predicted box, the mask of the same class is attached to the instance by adding a
        new "pred_masks" field to pred_instances.

        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            pred_instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. Each Instances must have field "pred_classes".

        Returns:
            None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
                Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
                masks the resolution predicted by the network; post-processing steps, such as resizing
                the predicted masks to the original image resolution and/or binarizing them, is left
                to the caller.
        """
        cls_agnostic_mask = pred_mask_logits.size(1) == 1

        if cls_agnostic_mask:
            #mask_probs_pred = pred_mask_logits.sigmoid()
            mask_probs_pred =torch.sigmoid(pred_mask_logits)
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_mask_logits.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            device = (
                class_pred.device
                if torch.jit.is_scripting()
                else ("cpu" if torch.jit.is_tracing() else class_pred.device)
            )
            indices = move_device_like(torch.arange(num_masks, device=device), class_pred)
            mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

        #mask_probs_pred = self.mask_aggregator(mask_probs_pred,pred_instances,training=False)
        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    
        for prob, instances in zip(mask_probs_pred, pred_instances):
            instances.pred_masks = prob  # (1, Hmask, Wmask)

def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
