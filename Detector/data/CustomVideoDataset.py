# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import itertools
import logging
import numpy as np
import pickle
import random
from typing import Callable, Union
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import os
from detectron2.utils.serialize import PicklableWrapper
from data.datasets.coco import load_coco_json
__all__ = ["CustomVideoDataset"]

logger = logging.getLogger(__name__)




class CustomVideoDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self,dataset, map_func,video_in_batch=1,image_in_video=8,is_train= True):
        self.dataset = dataset
        self._map_func = PicklableWrapper(map_func)
        self.video_dict = self._group_by_videos()
        self.video_ids = sorted(self.video_dict.keys())
        
        self.video_in_batch= int(video_in_batch)
        self.image_in_video =int(image_in_video)
        self.is_train = is_train
        
        if not self.is_train:
            self.segments = self._generate_segments()
            
    def _group_by_videos(self):
        video_dict = {}
        for d in self.dataset:
            video_id = d["video_id"]
            if video_id not in video_dict:
                video_dict[video_id] = []
            video_dict[video_id].append(d)
        
        
        return video_dict

    def _generate_segments(self):
        segments = []
        for video_id in self.video_ids:
            frames = sorted(self.video_dict[video_id], key=lambda x: x["frame_id"])
            num_frames = len(frames)
            for start in range(0, num_frames - self.image_in_video + 1, self.image_in_video):
                segments.append((video_id, start))
                #print((video_id, start))
        return segments
    '''
    def is_target_file(self,file_name):
        targets = [
        'train/ILSVRC2015_VID_train_0003/ILSVRC2015_train_01091000/',
        'train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00627000/000185.JPEG',
        'train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00627000/000186.JPEG',
        'train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00627000/000187.JPEG',
        ]
        return any(target in file_name for target in targets)
    '''
    def __len__(self):
        
        if self.is_train:
            
            return len(self.video_ids)
        else:
            
            return len(self.segments)
            

    def __getitem__(self, idx):
        if self.is_train:
            while True:
                '''
                videos = random.sample(self.video_dict,self.video_in_batch)
                for video in videos:
                    imgnpath = video.replace('/annotations/train_videos_json/','/train/')
                    dicts= load_coco_json(video,)
                '''
                
                video_ids = random.sample(list(self.video_dict.keys()), self.video_in_batch)
            #print('video_ids::',video_ids)
                images = []
                valid = True
                
                for video_id in video_ids:
                    frames = sorted(self.video_dict[video_id], key=lambda x: x["frame_id"])
                    if len(frames) < self.image_in_video:
                        valid = False
                        break
                    
                    start_idx = random.randint(0, len(frames) - self.image_in_video)
                    selected_frames = frames[start_idx:start_idx + self.image_in_video]
                
                    for frame in selected_frames:
                        # Check if segmentation exists and is non-empty
                        if not frame['annotations'] or not frame['annotations'][0].get('segmentation'):
                            valid = False
                            break
                        #if self.is_target_file(frame['file_name']):
                        #    valid = False
                        #    break
                    
                    if not valid:
                        break

                    for frame in selected_frames:
                        processed_frame = self._map_func(frame)
                        images.append(processed_frame)
                
                if valid:
                    return images
        else:
           
            video_id, start_idx = self.segments[idx]
            frames = sorted(self.video_dict[video_id], key=lambda x: x["frame_id"])
            selected_frames = frames[start_idx:start_idx + self.image_in_video]
            images = [self._map_func(f) for f in selected_frames]
            return images
            