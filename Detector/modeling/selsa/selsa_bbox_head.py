# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
from mmdet.models import  ConvFCBBoxHead
from .aggerator import SelsaAggregator
from mmdet.core import multi_apply
#@HEADS.register_module()
class SelsaBBoxHead(ConvFCBBoxHead):
    """Selsa bbox head.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        aggregator (dict): Configuration of aggregator.
    """

    def __init__(self,*args, **kwargs):
        	 
        super(SelsaBBoxHead, self).__init__(*args, **kwargs)
        self.aggregator = nn.ModuleList()
        
        #self.shared_fcs
        #print('##########')
        self.shared_fcs[0].in_features=4096
        #print(self.shared_fcs[0].weight.size())
        for i in range(self.num_shared_fcs):
            self.aggregator.append(SelsaAggregator(in_channels=1024,num_attention_blocks=16)).cuda()
        self.inplace_false_relu = nn.ReLU(inplace=False).cuda()
  
    def forward(self, x, ref_x):
        """Computing the `cls_score` and `bbox_pred` of the features `x` of key
        frame proposals.

        Args:
            x (Tensor): of shape [N, C, H, W]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C, H, W]. M is the number of reference
                frame proposals.

        Returns:
            tuple(cls_score, bbox_pred): The predicted score of classes and
            the predicted regression offsets.
        """
        # shared part
        
        
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
                ref_x = conv(ref_x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
                ref_x = self.avg_pool(ref_x)

            x = x.flatten(1)
            ref_x = ref_x.flatten(1)
            #print(x.size())
            #print('##########')
            #print(self.shared_fcs)
            for i, fc in enumerate(self.shared_fcs):
                #print(fc)
                #print(fc.weight.size(),fc.bias.size())
                x = fc(x)
                ref_x = fc(ref_x)
                x = x + self.aggregator[i](x, ref_x)
                ref_x = self.inplace_false_relu(ref_x)
                x = self.inplace_false_relu(x)

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                            cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            #labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight

            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights





    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    #gt_labels,
                    rcnn_train_cfg,
                    concat=True):

        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        #pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        label,label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            #pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            #labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return None,label_weights, bbox_targets, bbox_weights