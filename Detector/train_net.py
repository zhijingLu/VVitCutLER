#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py

"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import torch
from collections import OrderedDict
from data.datasets.coco import register_coco_instances
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from config import add_cutler_config
from detectron2.data import MetadataCatalog
from engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.engine import hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA
import data # register new datasets
import modeling.roi_heads

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    evaluator_type ='coco'
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    #print('evaluator_type:',evaluator_type)
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, no_segm=cfg.TEST.NO_SEGM))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def extract_and_save_features(cfg, model, dataset_name, feature_layer="res5", output_path="features.npy"):
    """
    遍历dataset_name，提取模型backbone指定层的特征，池化后保存成.npy

    参数:
        cfg: detectron2配置
        model: 已加载好的model.eval()状态
        dataset_name: 注册的数据集名称，如"ImageNet-VID-val"
        feature_layer: backbone输出字典中的键，默认"res5"
        output_path: npy保存路径
    """
    from data.build import build_detection_test_loader
    from data.dataset_mapper import DatasetMapper
    from detectron2.structures import ImageList, Instances
    import torch.nn.functional as F
    import numpy as np

    def pad_to_divisible(tensor, divisor=32):
        _, _, h, w = tensor.shape
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    model.eval()
    features = []
    labels = []
    mapper = DatasetMapper(cfg, is_train=False)
    dataloader = build_detection_test_loader(cfg, dataset_name,mapper=mapper)
    device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if count >= 10000:
                break
            images = [x["image"].to(device).float() / 255.0 for x in batch[0]]
            images = ImageList.from_tensors(images, model.backbone.size_divisibility)
 
            # 这里batch size=1更好，Detectron2的dataloader一般默认是1
            assert len(images) == 1
            img_tensor = images[0].unsqueeze(0)  # (1, C, H, W)
            img_tensor = pad_to_divisible(img_tensor, 32)
            #print('img_tensor',img_tensor.size())
            # backbone输出是dict，拿feature_layer对应的tensor
            feat_dict = model.backbone(img_tensor)
            if feature_layer not in feat_dict: 
                raise ValueError(f"Feature layer '{feature_layer}' not found in backbone output")
            feat_map = feat_dict[feature_layer]  # (1, C, H, W)

            # 全局池化，得到向量 (C,)
            pooled_feat = F.adaptive_avg_pool2d(feat_map, (1, 1)).squeeze().cpu().numpy()
            features.append(pooled_feat)

            # label可以从batch[0]["instances"].gt_classes中获得，如果有
            # 如果是检测任务，gt_classes可能是多标签，挑选第一个或者其他策略
            
            labels.append(1)  # 无标签时
            count+=1
            print(count)

    features = np.array(features)
    labels = np.array(labels)

    np.save(output_path.replace(".npy", "_features.npy"), features)
    np.save(output_path.replace(".npy", "_labels.npy"), labels)
    print(f"Feature and label saved to {output_path.replace('.npy', '_features.npy')} and _labels.npy")



class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        #print('into build_evaluator')
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # FIXME: brute force changes to test datasets and evaluation tasks
    if args.test_dataset != "": cfg.DATASETS.TEST = ((args.test_dataset),)
    if args.train_dataset != "": cfg.DATASETS.TRAIN = ((args.train_dataset),)
    cfg.TEST.NO_SEGM = args.no_segm
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    '''
    #DAVIS
    #jsonfile = '/netscratch/zlu/dataset/DAVIS/origin_anno/DAVIS_origin_train2_r1_thresh0.8.json'
    #jsonfile = '/netscratch/zlu/dataset/DAVIS/flow-raft/davis_raft_refine2_train.json'
    #jsonfile = '/netscratch/zlu/dataset/DAVIS/vit/sam_xml/DAVIS_vit_train.json'
    #jsonfile = '/netscratch/zlu/dataset/DAVIS/flow-raft/davis_raft_refine2_train_r1_thresh0.8.json'
    jsonfile = '/netscratch/zlu/dataset/DAVIS/origin_anno/DAVIS_origin_train2.json'
    image_root = '/netscratch/zlu/dataset/DAVIS/train/'
    register_coco_instances('DAVIS',{},jsonfile,image_root)
    #jsonfile_val='/netscratch/zlu/dataset/DAVIS/vit/sam_xml/DAVIS_vit_val.json'
    #jsonfile_val='/netscratch/zlu/dataset/DAVIS/flow-raft/davis_raft_refine2_val_r1_thresh0.8.json'
    #jsonfile_val='/netscratch/zlu/dataset/DAVIS/flow-raft/davis_raft_refine2_val.json'
    jsonfile_val='/netscratch/zlu/dataset/DAVIS/origin_anno/DAVIS_origin_val.json'
    #jsonfile_val='/netscratch/zlu/dataset/DAVIS/origin_anno/DAVIS_origin_val_r1_thresh0.8.json'
    val_image_root='/netscratch/zlu/dataset/DAVIS/val/'
    register_coco_instances('DAVIS-val',{},jsonfile_val,val_image_root)
    '''
    '''
    #youtubevos
    #jsonfile='/netscratch/zlu/dataset/youtubevos/cutvler_origin/cutvler_origin_anno/origin_val_total.json'
    #image_root='/netscratch/zlu/dataset/youtubevos/val/'
    jsonfile = '/netscratch/zlu/dataset/youtubevos/vit/sam_xml/ytvis_vit_train.json'
    #jsonfile = '/netscratch/zlu/cutLER-aglin/cutler/output/align_agg/ytvis/maskrcnn/ytvis_videocut_train_r1_thresh0.8.json'
    #jsonfile = '/netscratch/zlu/dataset/youtubevos/cutvler_origin/cutvler_origin_anno/origin_train_total.json'
    #jsonfile = '/netscratch/zlu/dataset/youtubevos/flow_refine/ytvos21_flow_train.json'
    #jsonfile = '/netscratch/zlu/dataset/youtubevos/maskcut/youtubevos_train_maskcut.json'
    #jsonfile = '/netscratch/zlu/dataset/youtubevos/predictor_anno/youtube21_train_samtrainedpredictor_1model.json'
    image_root = '/netscratch/zlu/dataset/youtubevos/train/JPEGImages/'
    register_coco_instances('youtubevos-train',{},jsonfile,image_root)
    #jsonfile_val='/netscratch/zlu/dataset/youtubevos/vit/sam_xml/ytvis_vit_val.json'
    #jsonfile_val='/netscratch/zlu/dataset/youtubevos/cutvler_origin/cutvler_origin_anno/origin_val_total.json'
    #jsonfile_val='/netscratch/zlu/dataset/youtubevos/flow_refine/ytvos21_flow_val.json'
    #jsonfile_val='/netscratch/zlu/dataset/youtubevos/maskcut/youtubevos_val_maskcut.json'
    jsonfile_val='/netscratch/zlu/dataset/ytvis_2021/val_gt.json'
    val_image_root='/netscratch/zlu/dataset/youtubevos/val/'
    register_coco_instances('youtubevos-val',{},jsonfile_val,val_image_root)
    '''
    ''' 
    #epic-kitchen
    jsonfile = '/netscratch/zlu/dataset/kitchen100/2v/GroundTruth-SparseAnnotations/annotations/kitchen100_train.json'
    image_root = '/netscratch/zlu/dataset/kitchen100/2v/GroundTruth-SparseAnnotations/rgb_frames/train/'
    register_coco_instances('epickitchen',{},jsonfile,image_root)
    jsonfile_val='/netscratch/zlu/dataset/kitchen100/2v/GroundTruth-SparseAnnotations/annotations/kitchen100_val.json'
    val_image_root='/netscratch/zlu/dataset/kitchen100/2v/GroundTruth-SparseAnnotations/rgb_frames/val/'
    register_coco_instances('epickitchen-val',{},jsonfile_val,val_image_root)
    '''
    
    #image-vid
    #jsonfile = '/netscratch/zlu/cutLER-aglin/cutler/output/align_agg/imagenetvid/maskrcnn/imagenetvid_videocut_train_r1_thresh0.8.json'
    #jsonfile = '/netscratch/zlu/dataset/imagenet-vid/flow/imagenetvid_flow_train.json'
    #jsonfile = '/netscratch/zlu/dataset/imagenet-vid/vit/sam_xml/Imagenetvid_vit_train.json'
    #image_root = '/netscratch/zlu/dataset/imagenet-vid/train/'
    jsonfile = '/netscratch/zlu/dataset/imagenet-vid/cutvleranno/origin_train.json'
    #jsonfile = '/netscratch/zlu/cutLER-aglin/cutler/output/align_agg/imagenetvid/maskrcnn/imagenetvid_votecut_train_r1_thresh0.8.json'
    image_root = '/netscratch/zlu/dataset/imagenet-vid/' 
    register_coco_instances('ImageNet-VID',{},jsonfile,image_root)
    #jsonfile_val='/netscratch/zlu/dataset/imagenet-vid/flow/imagenetvid_flow_val.json'
    #jsonfile_val='/netscratch/zlu/dataset/imagenet-vid/cutvleranno/origin_val.json'
    jsonfile_val='/netscratch/zlu/dataset/imagenet-vid/annotations/val_gt.json'
    #jsonfile_val='/netscratch/zlu/dataset/imagenet-vid/vit/sam_xml/Imagenetvid_vit_val.json'
    val_image_root='/netscratch/zlu/dataset/imagenet-vid/val/'
    register_coco_instances('ImageNet-VID-val',{},jsonfile_val,val_image_root)
    
    
    #image-vid sam
    #jsonfile = '/netscratch/zlu/dataset/imagenetvid_sam/annotations/trimmed_imagenet_vid.json'
    #jsonfile = '/netscratch/zlu/dataset/imagenetvid_sam/annotations/imagenet_vid_train_updated.json'
    #jsonfile = '/netscratch/zlu/dataset/imagenetvid_sam/annotations/imagenet_vid_train_sam.json'
    '''
    jsonfile = '/netscratch/zlu/dataset/imagenetvid_sam/annotations/train_sam2.json'
    
    #jsonfile = '/netscratch/zlu/dataset/imagenetvid_sam/annotations/imagenet_vid_train_sam.json'
    #jsonfile = '/netscratch/zlu/dataset/imagenet/annotations/imagenet_train_fixsize480_tau0.15_N3.json'
    image_root = '/netscratch/zlu/dataset/imagenetvid_sam/train'
    register_coco_instances('ImageNet-VID-sam',{},jsonfile,image_root)
    #jsonfile_val='/netscratch/zlu/dataset/imagenetvid_sam/annotations/val_output.json'
    jsonfile_val='/netscratch/zlu/dataset/imagenetvid_sam/annotations/val_sam2.json'
    #jsonfile_val = '/netscratch/zlu/dataset/imagenet/annotations/imagenet_valid_fixsize480_tau0.15_N3.json'
    #val_image_root='/netscratch/zlu/dataset/imagenet/val/'
    val_image_root = '/netscratch/zlu/dataset/imagenetvid_sam/val'
    register_coco_instances('ImageNet-VID-val',{},jsonfile_val,val_image_root)
    '''
    '''
    #ovis
    train_jsonfile = '/netscratch/zlu/dataset/ovis/annotations/ovis_train_fixsize480_tau0.15_N3.json'
    train_image_root = '/netscratch/zlu/dataset/ovis/train/'
    register_coco_instances('OVIS',{},train_jsonfile,train_image_root)
    jsonfile_val='/netscratch/zlu/dataset/ovis/annotations/ovis_val_fixsize480_tau0.15_N3.json'
    val_image_root = '/netscratch/zlu/dataset/ovis/valid/'
    register_coco_instances('OVIS-val',{},jsonfile_val,val_image_root)
    '''
    #print(args.eval_only)
    if args.eval_only:
        
        model = Trainer.build_model(cfg)
        #print('into build++++++++')
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        #print('load finish+++++++++++')
        res = Trainer.test(cfg, model)
        
        #print('#############',cfg.TEST.AUG.ENABLED)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        #print('........',comm.is_main_process())
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    '''
    extract_features= True
    if extract_features:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        output_path = os.path.join(cfg.OUTPUT_DIR, "ytvis_aggregate_features.npy")
        extract_and_save_features(cfg, model, dataset_name=cfg.DATASETS.TEST[0], feature_layer="p5", output_path=output_path)
        return
    '''
    
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # print(args)
    # args.opts = postprocess_args(args.opts)
    # rint = random.randint(0, 10000)
    # args.dist_url = args.dist_url.replace('12399', str(12399 + rint))
    #print("Command Line Args:", args)
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
