# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/__init__.py

#from .faster_head import RPN  # isort:skip
#from .build import PROPOSAL_GENERATOR_REGISTRY, build_proposal_generator
#from .rpn import RPN_HEAD_REGISTRY, build_rpn_head, RPN, StandardRPNHead
from .aggerator import SelsaAggregator
from .roi_extractor import SingleRoIExtractor
from .selsa_roi_head import SelsaRoIHead
from .selsa_bbox_head import SelsaBBoxHead
#from .max_iou_assigner import MaxIoUAssigner


__all__ = list(globals().keys())

