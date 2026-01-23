# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/roi_heads.py

import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import os
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, pairwise_iou
from structures import pairwise_iou_max_scores
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
import statistics 
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers,fast_rcnn_inference
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
#from .mask_head import build_mask_head
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.box_regression import Box2BoxTransform
import torch.nn.functional as F
from colored import fg

from detectron2.utils.visualizer import ColorMode, Visualizer,VisImage
from detectron2.data.detection_utils import read_image


blue, red = fg('blue'), fg('red')

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    #print('build_roi_heads',name)
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    #proposals = proposals.to('cuda')
    fg_proposals = []
    fg_selection_masks = []
    #print('bg_label',bg_label)
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        #print(fg_selection_mask.device,':',fg_idxs.device)
        fg_idxs =fg_idxs.to(fg_selection_mask.device) 
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of foreground classes (i.e. background is not included)
            batch_size_per_image (int): number of proposals to sample for training
            positive_fraction (float): fraction of positive (foreground) proposals
                to sample for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
        }

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        #print('++++')
        #print(gt_classes)
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
            #print('has_gt',gt_classes)
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            #print('no has',gt_classes)
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            #print('match_quality_matrix:',match_quality_matrix)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            #print('#####',targets_per_image.gt_classes)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            #print('num_bg_samples',num_bg_samples)
            #print('num_fg_samples',num_fg_samples)
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        res5: nn.Module,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        if isinstance(res5, (list, tuple)):
            res5 = nn.Sequential(*res5)
        self.res5 = res5
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods."
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )
        return ret

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            feature_list = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(feature_list, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
class CustomStandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        box2box_transform = Box2BoxTransform,
        use_droploss: bool = False,
        droploss_iou_thresh: float = 1.0,
        use_temporalloss:bool = False,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.batchindex = 0
        #print(',,,,,,',mask_in_features)

        self.mask_on = mask_in_features is not None
        #print(self.mask_on)
        
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head
        
        self.train_on_pred_boxes = train_on_pred_boxes
        self.use_droploss = use_droploss
        self.box2box_transform = box2box_transform
        self.droploss_iou_thresh = droploss_iou_thresh
        self.topk = 150
        self.use_temporalloss =use_temporalloss 

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if cfg.MODEL.ROI_HEADS.USE_DROPLOSS:
            ret['use_droploss'] = True
            ret['droploss_iou_thresh'] = cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH
        if cfg.MODEL.ROI_HEADS.USE_TEMPORALLOSS:
            ret['use_temporalloss'] = True
        ret['box2box_transform'] = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If CustomStandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        #print('##########cg::::',cfg.MODEL.MASK_ON)
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    @torch.no_grad()
    def visual_image(self,batchindex,images,img_filename,proposal,is_gt):
        root = '/netscratch/zlu/CutLER-main/cutler/output/small_dataset/top150_10aggreall_small/proposals/'
        box_size = min(len(proposal), 7)
        
        img = convert_image_to_rgb(images.permute(1, 2, 0), "RGB")
        #print('image:::',images.size())
        visualizer = Visualizer(img, None)
        #vis_output  = VisImage(image, scale=1.0)
        proposal=proposal[0:box_size].cpu().detach().numpy()
       #proposal = proposal.cpu().detach().numpy()
        
        vis_output = visualizer.overlay_instances(boxes=proposal)
        img_filename  = os.path.basename(img_filename).split('.')[0]
        batch_folder = os.path.join(root,str(batchindex))
        
        if not os.path.exists(batch_folder):
            os.mkdir(batch_folder)
        if is_gt == False:
            vis_output.save(os.path.join(batch_folder,(img_filename+'_aggre.jpg')))
        else:
            vis_output.save(os.path.join(batch_folder,(img_filename+'.jpg')))

    @torch.no_grad()        
    def label_and_sample_proposals_topk(
        self, proposals: List[Instances], targets: List[Instances],topk = 150
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            
            if proposals_per_image.has("objectness_logits"):
                objectness_logits = proposals_per_image.objectness_logits
                _, topk_indices = torch.topk(objectness_logits, k=min(topk, len(objectness_logits)), largest=True)
                proposals_per_image = proposals_per_image[topk_indices]
            
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            

            if has_gt:

                sampled_targets = matched_idxs[sampled_idxs]
                #print('sampled_targets',sampled_targets.size())
                
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                
                    
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.
            #print('has___:',proposals_per_image.gt_masks)

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            #print('num_bg_samples',num_bg_samples)
            #print('num_fg_samples',num_fg_samples)
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
    
    def forward(
        self,
        images,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        file_name: List[str],
        targets: Optional[List[Instances]] = None,
        
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        #print(images[0])
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals_topk(proposals, targets, self.topk)
            #proposals = self.label_and_sample_proposals(proposals, targets)
            ####################
            
            features = [features[f] for f in self.box_in_features]
            
            index_nogtmask = []
            filtered_proposals =[]
            for i in range(0,len(proposals)):
                if hasattr(proposals[i], 'gt_masks'):
                    filtered_proposals.append(proposals[i])
                else:
                    index_nogtmask.append(i)
            refine_feature= []
            for single_feature in features:
                singlefeature_list = [t for t in single_feature]
                singlefearure_refine =[]
                for j in range(0,len(singlefeature_list)):
                    if j not in index_nogtmask:
                        singlefearure_refine.append(singlefeature_list[j])
                single_feature = torch.stack(singlefearure_refine, dim=0) 
                refine_feature.append(single_feature)  
            features = refine_feature
            proposals = filtered_proposals
            
            ####################


        del targets

        if self.training:
            losses= self._forward_box(images,features, proposals,file_name)
            #losses,pre_instance = self._forward_box(images,features, proposals,file_name)
            
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask_test(features, proposals))
            #losses.update(self._forward_mask(features,proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(images,features, proposals,file_name)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            #print('pred_instances:::',pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
       
        instances = self._forward_mask_test(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances
    
    @torch.no_grad()  
    def chooseTopk_singleproposal(self,proposal,topk=150):
        #print('into ++++++++++++')
        proposal_boxes = proposal.proposal_boxes.tensor
        score = proposal.objectness_logits
        if topk is not None and len(proposal_boxes) < topk:
            topk = len(proposal_boxes)

        topk_scores, topk_indices = torch.topk(score, topk,largest=True,sorted=True)
        
        #sorted_indices = torch.argsort(score, descending=True)
        #topk_indices = sorted_indices[:topk]
        proposal_boxes = proposal_boxes[topk_indices]
        #print('proposal_boxes',proposal_boxes.size())
        #extended_box= torch.zeros(512, 4)
        #extended_box[:topk, :] = proposal_boxes
        proposal.proposal_boxes.tensor = proposal_boxes.to('cuda')
        #score_new = torch.full((512,), -100.0)
        #score_new[:topk] = topk_scores
        proposal.objectness_logits = topk_scores.to('cuda')
        #proposal.gt_classes =  proposal.gt_classes[:topk]
        #print(proposal)
        return proposal
    
    def temporal_iou_consistency_loss(self, predictions, proposals, num_videos, num_frames_per_video):
        
        scores_all, deltas_all = predictions
        proposal_boxes_all = [x.proposal_boxes.tensor for x in proposals]
        loss_temporal_iou = 0.0
        loss_temporal_cls = 0.0
        count = 0
        for v in range(num_videos):
            preds_list = []
            prop_boxes_list = []
            for f in range(num_frames_per_video):
                idx = v * num_frames_per_video + f
                start = idx * len(proposals[0].proposal_boxes.tensor)
                end = (idx + 1) * len(proposals[0].proposal_boxes.tensor)
                
                scores_frame = scores_all[start:end]
                deltas_frame = deltas_all[start:end]
                preds_list.append((scores_frame, deltas_frame))
                prop_boxes_list.append(proposal_boxes_all[idx])

        for t in range(num_frames_per_video - 1):
            scores_t, deltas_t = preds_list[t]
            
            scores_tp1, deltas_tp1 = preds_list[t + 1]

            props_t = prop_boxes_list[t]
            props_tp1 = prop_boxes_list[t + 1]

            # Calculate timing frame consistency:IoU
            boxes_t = self.box2box_transform.apply_deltas(deltas_t, props_t)
            boxes_tp1 = self.box2box_transform.apply_deltas(deltas_tp1, props_tp1)

            # Calculate IoU loss: maximum IoU between adjacent frames
            ious = pairwise_iou(Boxes(boxes_t), Boxes(boxes_tp1))
            max_iou_per_box, _ = ious.max(dim=1)
            loss_iou = (1.0 - max_iou_per_box).mean()
            loss_temporal_iou += loss_iou
            
            scores_t_fg = scores_t[:, 1]  # 1类是前景
            scores_tp1_fg = scores_tp1[:, 1]
            loss_cls = torch.nn.functional.mse_loss(scores_t_fg, scores_tp1_fg)
            loss_temporal_cls += loss_cls

            count += 1

    # 平均损失
        if count > 0:
            loss_temporal_iou /= count
        #loss_temporal_cls /= count
        #print('loss_temporal_iou',loss_temporal_iou)

        return loss_temporal_iou


    
    def _forward_box(self, images,features: Dict[str, torch.Tensor], proposals: List[Instances],file_name):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        #del file_name,images
        
        #features = [features[f] for f in self.box_in_features]
        

        if self.training:
            imgs_invideo=8
            assert features[0].size(0) ==len(proposals)
            proposal_boxes_tmp =Boxes.cat([x.proposal_boxes for x in proposals])
            if proposal_boxes_tmp.tensor.size(0) < (self.topk*len(proposals)):
                for i in range(0,len(proposals)):
                    if proposals[i].proposal_boxes.tensor.size(0) < self.topk:
                        pad_proposal = torch.zeros(self.topk,proposals[i].proposal_boxes.tensor.size(1)).to('cuda')
                        pad_proposal[:proposals[i].proposal_boxes.tensor.size(0), :] = proposals[i].proposal_boxes.tensor
                        proposals[i].proposal_boxes.tensor = pad_proposal.to('cuda')
                    if proposals[i].gt_classes.size(0) < self.topk:
                        pad_class = torch.ones(self.topk).to('cuda')
                        pad_class[:proposals[i].gt_classes.size(0)] = proposals[i].gt_classes
                        proposals[i].gt_classes= pad_class.long().to('cuda')
                    if proposals[i].objectness_logits.size(0) < self.topk:    
                        pad_objectness_logits = torch.full((self.topk,), -1, dtype=torch.float32)
                        pad_objectness_logits[:proposals[i].objectness_logits.size(0)] = proposals[i].objectness_logits
                        proposals[i].objectness_logits= pad_objectness_logits.to('cuda')
                    if proposals[i].gt_boxes.tensor.size(0) < self.topk:
                        pad_gt = torch.zeros(self.topk,proposals[i].gt_boxes.tensor.size(1)).to('cuda')
                        pad_gt[:proposals[i].gt_boxes.tensor.size(0), :] = proposals[i].gt_boxes.tensor
                        proposals[i].gt_boxes.tensor = pad_gt
            #proposals_num_list = [len(x.gt_boxes) for x in proposals]
            #print('proposals:',proposals_num_list)
            #print('++++',len(proposals))

            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals]) # torch.Size([topk* batch_size, 256, 7, 7])           
            #print(box_features.size())
            box_features = self.box_head(box_features) # torch.Size([topk * batch_size, 1024])  
            #print(box_features.size())
            predictions = self.box_predictor(box_features,0,1,True,self.topk,imgs_invideo)
            #print(proposals.size)
            #show the previous box
            '''
            vis = False
            if vis == True and self.batchindex % 1000 ==0 :
                _,prebox= self.box_predictor(box_features,0,0,True,self.topk,len(proposals))
                #visualize                
                num_prop_per_image = [len(p) for p in proposals]
                pre=prebox.split(num_prop_per_image) 
                for i in range(0,len(num_prop_per_image)): 
                    predictions_bbox_tmp_pre = self.box2box_transform.apply_deltas(pre[i], proposals[i].proposal_boxes.tensor) 
                    self.visual_image(self.batchindex,images[i],file_name[i],predictions_bbox_tmp_pre,True)
             
            #predictions = self.box_predictor(box_features,0,1,True) # [torch.Size([512 * batch_size, 2]), torch.Size([512 * batch_size, 4])]
            if vis == True and self.batchindex % 1000 ==0 :
                afte=predictions[1].split(num_prop_per_image) 
                for i in range(0,len(num_prop_per_image)):
                    predictions_bbox_tmp = self.box2box_transform.apply_deltas(afte[i], proposals[i].proposal_boxes.tensor) 
                    self.visual_image(self.batchindex,images[i],file_name[i],predictions_bbox_tmp,False)
            '''
            
            
        if not self.training:
            imgs_invideo=8
            '''
            num_prop_per_image = [len(p) for p in proposals]
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals]) # torch.Size([512 * batch_size, 256, 7, 7])
            box_features = self.box_head(box_features) # torch.Size([512 * batch_size, 1024])
            score,_= self.box_predictor(box_features,0,0,False,self.topk)
            scores=score.split(num_prop_per_image)         
            i= 0
            for x in proposals:
                x_new = []
                score_list = []
                avg=0
                for w in range(0,scores[i].size(0)):
                    score_list.append(scores[i][w][0].item())
                score_list = sorted(score_list)              
                mean_score = score_list[int(len(score_list)*0.6)]
                #mean_score = statistics.median(score_list) 
                for j in range(0,scores[i].size(0)):
                    if scores[i][j][0] > mean_score:
                        x_new.append(x.proposal_boxes.tensor[j])
                   
                x.proposal_boxes.tensor = torch.stack(x_new)
                i+=1
               
            
            
            proposal_boxes_tmp =Boxes.cat([x.proposal_boxes for x in proposals])
            if proposal_boxes_tmp.tensor.size(0) < (512*len(proposals)):
                for i in range(0,len(proposals)):
                    if proposals[i].proposal_boxes.tensor.size(0) < 512:
                        pad_proposal = torch.zeros(512,proposals[i].proposal_boxes.tensor.size(1)).to('cuda')
                        pad_proposal[:proposals[i].proposal_boxes.tensor.size(0), :] = proposals[i].proposal_boxes.tensor
                        proposals[i].proposal_boxes.tensor = pad_proposal
        
            '''
            #print(proposals[0])
            proposals_new = []
            for i in range(0,len(proposals)):
                pro= self.chooseTopk_singleproposal(proposals[i],topk = self.topk)
                proposals_new.append(pro)
            proposals = proposals_new
            
            features = [features[f] for f in self.box_in_features]


            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals]) # torch.Size([512 * batch_size, 256, 7, 7])
            box_features = self.box_head(box_features) # torch.Size([512 * batch_size, 1024])
            #print('box_features',box_features.size()) 
            predictions = self.box_predictor(box_features,0,1,False, self.topk,imgs_invideo)
            
            
            vis = False
            if vis == True and self.batchindex % 1000 ==0 :
                _,prebox= self.box_predictor(box_features,0,0,False,self.topk,imgs_invideo)
                #visualize                
                num_prop_per_image = [len(p) for p in proposals]
                pre=prebox.split(num_prop_per_image) 
                for i in range(0,len(num_prop_per_image)): 
                    predictions_bbox_tmp_pre = self.box2box_transform.apply_deltas(pre[i], proposals[i].proposal_boxes.tensor) 
                    self.visual_image(self.batchindex,images[i],file_name[i],predictions_bbox_tmp_pre,True)
            
            #print('into aggre')
            #predictions = self.box_predictor(box_features,0,1,False)
            if vis == True and self.batchindex % 1000 ==0 :
                afte=predictions[1].split(num_prop_per_image) 
                for i in range(0,len(num_prop_per_image)):
                    predictions_bbox_tmp = self.box2box_transform.apply_deltas(afte[i], proposals[i].proposal_boxes.tensor) 
                    self.visual_image(self.batchindex,images[i],file_name[i],predictions_bbox_tmp,False)
            
	    

            
        self.batchindex = self.batchindex + 1  
        no_gt_found = False
        if self.use_droploss and self.training:
            # the first K proposals are GT proposals
            try:
                box_num_list = [len(x.gt_boxes) for x in proposals]
                #print('box_num_list:',box_num_list)
                gt_num_list = [torch.unique(x.gt_boxes.tensor[:100], dim=0).size()[0] for x in proposals]
                #print('gt_num_list:',gt_num_list)
            except:
                box_num_list = [0 for _ in proposals]
                gt_num_list = [0 for _ in proposals]
                no_gt_found = True

        if self.use_droploss and self.training and not no_gt_found:
            # NOTE: maximum overlapping with GT (IoU)
            predictions_delta = predictions[1]
            proposal_boxes =Boxes.cat([x.proposal_boxes for x in proposals])
            predictions_bbox = self.box2box_transform.apply_deltas(predictions_delta, proposal_boxes.tensor) #[512*n,4]
            idx_start = 0
            
            iou_max_list = []
            for idx, x in enumerate(proposals):
                idx_end = idx_start + box_num_list[idx]
                iou_max_list.append(pairwise_iou_max_scores(predictions_bbox[idx_start:idx_end], x.gt_boxes[:gt_num_list[idx]].tensor))
                idx_start = idx_end
            iou_max = torch.cat(iou_max_list, dim=0)
            #print('iou_max:',iou_max.size())
            #####test topk ,didn't change anything before####
            '''
            weight_list=[]
            for idx, x in enumerate(proposals):
                idx_end = idx_start + box_num_list[idx]
                weight_each=self.filter_max(pairwise_iou_max_scores(predictions_bbox[idx_start:idx_end], x.gt_boxes[:gt_num_list[idx]].tensor),topk_per_img=10)
                if weight_each.size(0) <self.topk:
                    #print(weight_each.size())
                    
                    temp=torch.zeros(self.topk)
                    temp[:weight_each.size(0)] = weight_each
                    weight_each = temp
                weight_list.append(weight_each)
                idx_start = idx_end
            weights = torch.cat(weight_list, dim=0)
            
            '''
            #########
        del box_features
        
        if self.training:
            if self.use_droploss and not no_gt_found:
                
                weights = iou_max.le(self.droploss_iou_thresh).float()
                
                weights = 1 - weights.ge(1.0).float()
               
                losses = self.box_predictor.losses(predictions, proposals, weights=weights.detach())
            else:
                losses = self.box_predictor.losses(predictions, proposals)

            #if self.use_temporalloss:

            #    n_videos= int(len(gt_num_list)/imgs_invideo)
            #    loss_temporal_iou = self.temporal_iou_consistency_loss(
            #                predictions, proposals, num_videos=n_videos, num_frames_per_video=imgs_invideo
            #            )
            if self.train_on_pred_boxes: # default is false
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            
            #########################
            #pred_instances, _ = self.box_predictor.proposal_topK(predictions, proposals,file_name)
            #return losses,pred_instances
            #########################
            del predictions,proposals
            return losses
        else:   
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            
            return pred_instances
    
    def _forward_mask_test(self, features: Dict[str, torch.Tensor],instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        
        if not self.mask_on:    
            return {} if self.training else instances
        #print('into mask++++++++++',instances[0])
        
        if self.training:
            # head is only trained on positive proposals.
            #number_per_instance = []
            #instances=instances.to('cuda')
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            #print(instances[0].proposal_boxes.tensor.size(),instances[1].proposal_boxes.tensor.size(),instances[3].proposal_boxes.tensor.size())
            #for instance in instances:
            #    number_per_instance.append(instance.proposal_boxes.tensor.size(0))
            
        
        if self.mask_pooler is not None:
            if self.training== False:
                features = [features[f] for f in self.mask_in_features]
                
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
            
            
        else:
            features = {f: features[f] for f in self.mask_in_features}
            #print(features)
        return self.mask_head(features,instances)

    def _forward_mask(self, features: Dict[str, torch.Tensor],instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        
        if not self.mask_on:    
            return {} if self.training else instances
        #print('into mask++++++++++',instances[0])
        
        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            
            features = self.mask_pooler(features, boxes)

        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def filter_max(self,iou_max,topk_per_img):
        
        topk_iou, topk_indices = torch.topk(iou_max, topk_per_img)
        
        weights = torch.zeros_like(iou_max)
        weights[topk_indices] = 1.0
        return weights



    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)
