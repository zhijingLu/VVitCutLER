# Copyright (c) OpenMMLab. All rights reserved.
#from mmdet.structures.bbox import bbox2result#, 
from mmdet.models import  StandardRoIHead
#from roi_heads import ROI_HEADS_REGISTRY
from .selsa_bbox_head import SelsaBBoxHead
from .roi_extractor import SingleRoIExtractor
import torch
#from .import 
from mmdet.core import RandomSampler,MaxIoUAssigner,bbox2result,bbox2roi
'''
def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        bboxes=bboxes
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois
'''


class SelsaRoIHead(StandardRoIHead):
    """selsa roi head."""
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        
        self.bbox_assigner = MaxIoUAssigner(pos_iou_thr=0.7,neg_iou_thr=0.3, min_pos_iou=0.3,ignore_iof_thr=-1)
        self.bbox_sampler = RandomSampler(num=256,pos_fraction=0.5, neg_pos_ub=-1,add_gt_as_proposals=False)
            
            
    def init_bbox_head(self):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        roi_layer=dict(type='RoIAlign', output_size=4, sampling_ratio=2)
        out_channels=256
        featmap_strides=[32]
        self.bbox_roi_extractor =SingleRoIExtractor(roi_layer=roi_layer,out_channels=out_channels,featmap_strides=featmap_strides)
        self.bbox_head = SelsaBBoxHead(num_shared_fcs=2,roi_feat_size=4)
    def forward(self,
                      x,
                      ref_x,
                      img_metas,
                      proposal_list,
                      ref_proposal_list,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            ref_x (list[Tensor]): list of multi-level ref_img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            ref_proposal_list (list[Tensors]): list of region proposals
                from ref_imgs.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #print('xxxxxxxxx',x.size())
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            #print('xxxxxxxxxxxxx')
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                #print(gt_bboxes[i].tensor.size())
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i].tensor,None,gt_bboxes_ignore[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i].tensor,
                    None,
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            #print('############')
            bbox_results = self._bbox_forward_train(x, ref_x, sampling_results,
                                                    ref_proposal_list,
                                                    gt_bboxes, None)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, ref_x, rois, ref_rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        #print('xxxxxx',x.size(),x[0].size())
        bbox_feats = self.bbox_roi_extractor(
            x,
            rois,
            ref_feats=ref_x)

        ref_bbox_feats = self.bbox_roi_extractor(
            ref_x, ref_rois)
        #print('exxaxxxx:',bbox_feats.device,ref_bbox_feats.device)
        
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            ref_bbox_feats = self.shared_head(ref_bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, ref_bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, ref_x, sampling_results,
                            ref_proposal_list, gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training."""
        
        rois = bbox2roi([res.bboxes for res in sampling_results])
        
        ref_rois = bbox2roi(ref_proposal_list)
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test(self,
                    x,
                    ref_x,
                    proposals_list,
                    ref_proposals_list,
                    #img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        
        #self.with_bbox=True
        #assert self.with_bbox, 'Bbox head must be implemented.'
        #print('###########',proposals_list)
        
        det_bboxes, det_labels = self.simple_test_bboxes(
            x,
            ref_x,
            proposals_list,
            ref_proposals_list,
            #img_metas,
            self.test_cfg,
            rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        #if not self.with_mask:
        return bbox_results
        '''
        else:
            mask_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, mask_results))
        '''

    def simple_test_bboxes(self,
                           x,
                           ref_x,
                           proposals,
                           ref_proposals,
                           #img_metas,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        #print('############')
        #print(proposals.type())
        rois = bbox2roi(proposals)
        ref_rois = bbox2roi(ref_proposals)
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois)
        #img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        #scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        bbox_pred = bbox_pred.split(
            num_proposals_per_img,
            0) if bbox_pred is not None else [None, None]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                None,
                1,
                #img_shapes[i],
                #scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
