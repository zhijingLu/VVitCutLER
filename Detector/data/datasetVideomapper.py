# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/dataset_mapper.py

import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

import data.detection_utils as utils
import data.transforms as T
from data.dataset_mapper import DatasetMapper
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetVideoMapper"]


class DatasetVideoMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        num_frames_per_video=4,
        num_videos_per_batch=4,
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        self.num_frames_per_video = num_frames_per_video
        self.num_videos_per_batch = num_videos_per_batch
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    
        #dataset_dict["instances"] = instances

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        batched_dataset = []
        
        for _ in range(self.num_videos_per_batch):
            
            video_idx = random.randint(0, len(dataset_dict) - 1)
            video_info = dataset_dicts[video_idx]

            # Collect frames for the selected video
            selected_frames = []
            frame_ids = set()

            for _ in range(self.num_frames_per_video):
                frame_idx = random.randint(0, len(video_info["annotations"]) - 1)
                frame_id = video_info["annotations"][frame_idx]["frame_id"]

                # Ensure frame_id is unique within the video
                while frame_id in frame_ids:
                    frame_idx = random.randint(0, len(video_info["annotations"]) - 1)
                    frame_id = video_info["annotations"][frame_idx]["frame_id"]

                frame_ids.add(frame_id)
                selected_frames.append(video_info["annotations"][frame_idx]["file_name"])

            # Create a new record for the batched dataset
            batched_record = {
                "video_id": video_info["video_id"],
                "frames": selected_frames
            }
            batched_dataset.append(batched_record)

        # Call parent method to handle image loading and augmentation
        return super().__call__(batched_dataset)