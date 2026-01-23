import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re

from pathlib import Path
def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,split: str = None) -> List[str]:

    # If use annotaion ids list
    #ext_with_dot = '.' + ext if ext != '' else ''
    ann_paths =[]
    #i=0
    if split == "train":
        for videoname in os.listdir(ann_dir_path):
            videopath=os.path.join(ann_dir_path,videoname)
            
            #print('videoname',videoname)
            for foldername in os.listdir(videopath):
                folderpath = os.path.join(videopath,foldername)
                for filename in os.listdir(folderpath):
                    annpath = os.path.join(folderpath,filename)
                    #print(annpath)
                    ann_paths.append(annpath)
    elif split == "val":
        #i=0
        #ref='/netscratch/zlu/dataset/imagenetvid_sam/dy_val_anno/val/'
        for foldername in os.listdir(ann_dir_path):
            folderpath = os.path.join(ann_dir_path,foldername)
            #i=i+1
            for filename in os.listdir(folderpath):
                    #if filename in files_in_ref:
                annpath = os.path.join(folderpath,filename)
                ann_paths.append(annpath)       
    #ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(foldername,annotation_root,img_type, extract_num_from_imgid=True):
    #folder = annotation_root.findtext('folder')
    
    #video_id1 = folder.split('/')[0].split('_')[-1]
    #video_id2 = folder.split('/')[-1].split('_')[-1]
    
    #video_id = folder.split('/')[-1]
    video_id =folder =foldername
    imgname = annotation_root.findtext('filename')
    #img_name =os.path.join(folder,(imgname+'.JPEG')) 
    img_name =os.path.join(folder,(imgname+'.'+img_type))
    #print('img_name',img_name)
    frame_id = imgname
    #frame_id = img_name.split('.')[0].split('_')[-1]
    #frame_id =imgname.split('/')[-1].split('.')[0]
    #print('frame_id.....',frame_id)
    if extract_num_from_imgid and isinstance(frame_id, str):
        frame_id = int(re.findall(r'\d+', frame_id)[0])
    video_id = ''.join(filter(str.isdigit, video_id))
    img_id =int(video_id + str(frame_id))
    video_id = int(video_id)
    #print(img_id,':::videoid:',video_id)
    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': img_name,
        'height': height,
        'width': width,
        'id': img_id,
        'video_id': video_id,
        'frame_id':frame_id
    }
    return image_info

def get_coco_video(ann_dir_path: str = None,split: str = None):
    video=[]
    #print('++++++++',ann_dir_path)
    if split == 'train':
        for videofirstname in os.listdir(ann_dir_path):
            #video_id_father = videofirstname.split('_')[-1]
            videofolder = os.path.join(ann_dir_path,videofirstname)
        for videoname in os.listdir(videofolder):
            #print('videoname',videoname)
            id = int(videoname.split('_')[-1])
            #print('id,',id)
            videoname_true = videoname
            video_single = {
            'id': id,
            'name': videoname_true
            }
            video.append(video_single)
            
    elif split == 'val':
        for videoname in os.listdir(ann_dir_path):
            id = int(videoname.split('_')[-1])
            #print('id,',id)
            videoname_true = videoname
            video_single = {
            'id': id,
            'name': videoname_true
            }
            video.append(video_single)
    return video
def get_coco_annotation_from_obj_bbox_only(obj):
    """
    仅生成 bbox，不处理 mask/segmentation
    """
    category_id = 1  # 默认类别，如果有 label2id 可以替换
    bndbox = obj.find('bndbox')
    xmin = float(bndbox.findtext('xmin'))
    ymin = float(bndbox.findtext('ymin'))
    xmax = float(bndbox.findtext('xmax'))
    ymax = float(bndbox.findtext('ymax'))
    
    #assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    if o_width <= 0 or o_height <= 0:
        return None

    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'segmentation': []  # COCO 格式要求字段存在，但为空即可
    }
    return ann

def get_coco_annotation_from_obj(obj):
    #label = obj.findtext('name')
    #assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = 1
    
    bndbox = obj.find('bndbox')
    xmin = float(bndbox.findtext('xmin')) 
    ymin = float(bndbox.findtext('ymin')) 
    xmax = float(bndbox.findtext('xmax'))
    ymax = float(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    
    if obj.find('mask') != None:
        mask = obj.find('mask').text
        mask_point = mask.split(';')
        masks = []
        for item in mask_point:
            if item =='':
               continue
            item=item.split(',')
            itemlist=list(map(float,item))
            masks.append(itemlist)
    else:
        mask = None
    
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        #'ignore': 0,
        'segmentation': masks  
    }
    return ann

def convert_xmls_to_cocojson(annotation_paths: List[str],
                             output_jsonpath: str,
                             ann_dir_path:str,
                             extract_num_from_imgid: bool = True,
                             img_type: str=None,
                             split: str = None,
                             ):
    output_json_dict = {
        "images": [],
        "video" : [],
        "annotations": [],
        "categories": []
    }
    print('Start converting !')

    #video = get_coco_video(ann_dir_path=ann_dir_path,split= split)
    #output_json_dict['video']= video
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        #print(a_path)
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()
        path = Path(a_path)
        folder_name = path.parts[-2] 
        img_info = get_image_info(folder_name,annotation_root=ann_root,img_type=img_type,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)
        for obj in ann_root.findall('object'):
            '''
            ann = get_coco_annotation_from_obj_bbox_only(obj=obj)
            if ann is not None:
                ann.update({'image_id': img_id, 'id': bnd_id})
                output_json_dict['annotations'].append(ann)
                bnd_id += 1

            '''
            if  obj.find('mask') != None:
                ann = get_coco_annotation_from_obj(obj=obj)
                ann.update({'image_id': img_id, 'id': bnd_id})
                output_json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1
            
    #for label, label_id in label2id.items():
    category_info = {'id': 1, 'name': 'pan'}
    output_json_dict['categories'].append(category_info)
    
    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)
    print('write json finish......')

    

def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default='/netscratch/zlu/dataset/imagenet-vid/annotations/smallxml/',
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Split to use")
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='/netscratch/zlu/dataset/imagenet-vid/annotations/val_gt_small.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    parser.add_argument("--img_type", type=str, default="JPEG", choices=["JPEG", "jpg"], help="Split to use")
    
    args = parser.parse_args()
    #label2id = get_label2id(labels_path=args.labels)
    ann_paths = get_annpaths(ann_dir_path=args.ann_dir,split=args.split)
    
    
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        output_jsonpath=args.output,
        ann_dir_path=args.ann_dir,
        extract_num_from_imgid=True,
        img_type=args.img_type,
        split=args.split
    )
    
    
    
    

if __name__ == '__main__':
    
    main()

