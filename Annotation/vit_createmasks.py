import os
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torch
from PIL import Image
import time
from helper.vitCut import get_Boxes,roi_align_on_image,mask2polygon,map_masks_to_original_image,warp_bbox_by_flow
from utils.annotaions_worker import CocoAnnotationsWorker
import json
from collections import defaultdict
from helper.write_xml import write_to_xml_vit,write_to_xml_box
import numpy as np
from transformers import AutoModel, AutoImageProcessor
from student_decoder.vit_sam2_model import DINOv2SegDecoder
import cv2


def get_adjacent_images(img_file,reference_count =1,image_type='jpg'):
    #reference_count =1 means get 1 previous image and 1 next image
    video_folder, image_number = parse_image_path(img_file)
    #print('#####',image_number)
    #video_folder = video_folder.replace('/train/JPEGImages/','/dyscore_anno/JPEGImages_small/')
    video_folder = Path(video_folder)
    image_indices = sorted([f.stem for f in video_folder.iterdir() if f.is_file()])
    
    
    if image_number not in image_indices:
        raise ValueError("Image not found in the specified video.")
    
    idx = image_indices.index(image_number)
    
    if idx == 0:  # first frame
        prev_images = []
        next_images = [
            (str(video_folder)+'/'+image_indices[i]+'.'+image_type) for i in range(idx + 1, min(len(image_indices), idx + 1 + reference_count))
        ]
        
        
    elif idx == len(image_indices) - 1:  # final frame
        prev_images = [
            (str(video_folder)+'/'+image_indices[i]+'.'+image_type) for i in range(max(0, idx - reference_count), idx)
        ]
        next_images = []
        
    else:  
        prev_images = [
            (str(video_folder)+'/'+image_indices[i]+'.'+image_type) for i in range(max(0, idx - reference_count), idx)
        ]
        #next_images = []
        
        next_images = [
            (str(video_folder)+'/'+image_indices[i]+'.'+image_type) for i in range(idx + 1, min(len(image_indices), idx + 1 + reference_count))
        ]
    framelist = prev_images + next_images
    #print(framelist)
    return framelist


def load_eig_vecs(eigenvec_dirs, num_eig_vecs, image_name):
    """
    Load the eigen vectors for the image in the format of a dictionary votecut method expects
    :param eigenvec_dirs: list of directories containing the eigen vectors
    :param num_eig_vecs: number of eigen vectors to use from each directory
    :param image_name: name of the image without the extension
    :return:
    """
    # load eigen vectors
    vector_groups = {}
    for eigenvec_dir in eigenvec_dirs:
        
        eig_vec_path = os.path.join(eigenvec_dir, f"{image_name}.pt")
        #print(eig_vec_path)
        eig_vec = torch.load(eig_vec_path)
        eig_vec = eig_vec.T
        eig_vec = eig_vec[:num_eig_vecs]
        if eig_vec.shape[1] == 900:
            eig_vec = eig_vec.reshape(eig_vec.shape[0], 30, 30)
        elif eig_vec.shape[1] == 3600:
            eig_vec = eig_vec.reshape(eig_vec.shape[0], 60, 60)
        elif eig_vec.shape[1] == 1156:
            eig_vec = eig_vec.reshape(eig_vec.shape[0], 34, 34)
        else:
            raise ValueError("Invalid eig vec shape")
        vector_groups[eigenvec_dir] = {
            "eigenvectors": eig_vec
        }
    return vector_groups


def parse_image_file(image_full_path):
    #print('image_name origin:',Path(image_full_path))
    frame_name = Path(image_full_path).stem.split(".")[0]
    video_name = Path(image_full_path).parent.name.split('_')[-1]
    if args.split == "train":
        image_name = "/".join(image_full_path.split("/")[-3:])
    else:
        image_name = "/".join(image_full_path.split("/")[-2:])
    image_name = image_name.split('.')[0]
    
    video_numbers = ''.join(filter(str.isdigit, video_name))
    if not isinstance(frame_name, int):
        frame_name= frame_name.split('_')[-1]
    #print('video_name::',video_name,'frame_name:',frame_name)
    # in case the file name starts with ILSVRC2012 remove it, it is the validation prefix
    #image_id = image_name[len("ILSVRC2015"):] if image_name.startswith("ILSVRC2015") else image_name
    image_id = int(video_numbers+frame_name)
    
    #print('image_id:',image_id)
    return image_name, image_id

def parse_image_path(image_path):
    
    current_video =  os.path.dirname(image_path)
    image_number = Path(image_path).stem.split(".")[0] #.split('_')[-1]
    #print('image_number::',image_number)
    return current_video, image_number

def compute_iou(box1, box2):
    """计算两个bounding box的IoU
    box = [x_min, y_min, x_max, y_max]
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x1 = max(x1, x3)
    y1 = max(y1, y3)
    x2 = min(x2, x4)
    y2 = min(y2, y4)
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area
     
    return intersection_area / union_area if union_area > 0 else 0
    

def merge_boxes(box1, box2, method="union"):
  
    if method == "union":
       
        x_min = min(box1[0], box2[0])
        y_min = min(box1[1], box2[1])
        x_max = max(box1[2], box2[2])
        y_max = max(box1[3], box2[3])
    elif method == "average":

        x_min = (box1[0] + box2[0]) / 2
        y_min = (box1[1] + box2[1]) / 2
        x_max = (box1[2] + box2[2]) / 2
        y_max = (box1[3] + box2[3]) / 2
    else:
        raise ValueError("Unknown merge method")

    return [x_min, y_min, x_max, y_max]

def fuse_bboxes(current_boxes, reference_boxes, iou_threshold=0.6, merge_method="average"):
    all_boxes = [list(box) for box in current_boxes + reference_boxes]
    fused_boxes = []

    while all_boxes:
        base = all_boxes.pop(0)
        i = 0
        while i < len(all_boxes):
            iou = compute_iou(base, all_boxes[i])
            if iou >= iou_threshold:
                base = merge_boxes(base, all_boxes.pop(i), method=merge_method)
                i = 0  # 重新遍历，因为 base 改变了
            else:
                i += 1
        fused_boxes.append(base)

    '''
    fused_boxes = []
    print('current_boxes',current_boxes)
    print('reference_boxes',reference_boxes)
    used_ref_boxes =[]
    for cur_box in current_boxes:
        for ref_box in reference_boxes:
            iou = compute_iou(cur_box, ref_box)
            if iou > iou_threshold:
                cur_box = merge_boxes(cur_box, ref_box)
                if ref_box not in used_ref_boxes:
                    used_ref_boxes.append(ref_box)
        fused_boxes.append(cur_box)
    '''
   
    return fused_boxes

def filter_current_by_reference(current_boxes, reference_boxes, iou_threshold=0.5):
    # 确保都是 list 类型
    reference_boxes = [list(box) for box in reference_boxes]
    current_boxes = [list(box) for box in current_boxes]
    #print('current_boxes',current_boxes)
    #print('reference_boxes',reference_boxes)
    filtered = []
    for cur_box in current_boxes:
        for ref_box in reference_boxes:
            iou = compute_iou(cur_box, ref_box)
            if iou >= iou_threshold:
                filtered.append(cur_box)
                break  # 一旦满足一个，就保留，不用检查剩下的
    return filtered
    
def fuse_boxes_greedy(current_boxes, ref_boxes, iou_threshold=0.7,current_iou_threshold=0.6):
    """
    基于 IoU 的贪心融合
    """
    min_group_size=3
    #all_boxes = [list(b) for b in current_boxes + ref_boxes]
    all_boxes=ref_boxes
    fused_boxes = []

    while all_boxes:
        base_box = all_boxes.pop(0)
        group = [base_box]
        remove_idx = []
        # 找到与 base_box IoU >= 阈值的框
        for i, box in enumerate(all_boxes):
            if compute_iou(base_box, box) >= iou_threshold:
                group.append(box)
                remove_idx.append(i)
        # 删除已经加入 group 的框
        for idx in reversed(remove_idx):
            all_boxes.pop(idx)
        if len(group) < min_group_size:
            continue

        # 融合 group
        fused_box = [
            max(int(sum(b[0] for b in group) / len(group)), 0),  # 平均 x1
            max(int(sum(b[1] for b in group) / len(group)), 0),  # 平均 y1
            int(sum(b[2] for b in group) / len(group)),          # 平均 x2
            int(sum(b[3] for b in group) / len(group))           # 平均 y2
        ]
        fused_boxes.append(fused_box)
    filtered_current = []
    for cbox in current_boxes:
        for fbox in fused_boxes:
            if compute_iou(cbox, fbox) >= current_iou_threshold:
                filtered_current.append(cbox)
                break  # 已匹配上一个就够了
    final_result = filtered_current[:]
    for fbox in fused_boxes:
        # 检查是否已有 current 匹配
        matched = any(compute_iou(cbox, fbox) >= iou_threshold for cbox in filtered_current)
        if not matched:
            final_result.append(fbox)
    return final_result

def fuse_with_reference(current_boxes, reference_boxes_list, iou_threshold=0.7, group_iou_threshold=0.7):
    """
    current_boxes: 当前帧的 bbox 列表
    reference_boxes_list: 多个参考帧的 bbox，每一帧是一个列表
    iou_threshold: 当前帧与参考框匹配的阈值
    group_iou_threshold: 参考帧之间 bbox 相似度（视为同一个目标）阈值
    """
    
    current_boxes = [list(b) for b in current_boxes]
    all_ref_boxes = [list(box) for box in reference_boxes_list]
    #print('current_boxes',current_boxes)
    #print('all_ref_boxes',all_ref_boxes)
    # 1. 聚合参考框：将参考帧中相似的 bbox 归为一类（group）
    groups = []
    used = [False] * len(all_ref_boxes)

    for i, box_i in enumerate(all_ref_boxes):
        if used[i]:
            continue
        group = [box_i]
        used[i] = True
        for j in range(i + 1, len(all_ref_boxes)):
            if not used[j] and compute_iou(box_i, all_ref_boxes[j]) >= group_iou_threshold:
                group.append(all_ref_boxes[j])
                used[j] = True
        groups.append(group)
   
    final_boxes = current_boxes[:]

    # 3. 检查每个 group 是否已被当前帧中的框匹配
    matched_current_boxes = set()
    for group in groups:
        matched = False
        for cur_box in current_boxes:
            if any(compute_iou(cur_box, ref_box) >= iou_threshold for ref_box in group):
                matched = True
                break  # 有一个匹配就行
        if not matched and len(group) > 1:
            # 当前帧中没人匹配这个 group，执行融合
            fused_box = [
                sum([box[0] for box in group]) / len(group),
                sum([box[1] for box in group]) / len(group),
                sum([box[2] for box in group]) / len(group),
                sum([box[3] for box in group]) / len(group),
            ]
            final_boxes.append(fused_box)

    return final_boxes



def create_ViTcut_annotations(eigenvec_dirs, img_files, Ks, worker_dir,
                               tau_m=0.2, num_eig_vecs=1, save_period=100, device="cuda", resume=False,vit=None,processor = None,decoder=None,image_type = 'jpg'):
    """
    This is a method for a single job that creates the pseudo labels for the images using votecut method and save them
    to a temporary file in order to be aggregated later. That way we can parallelize the process of creating the pseudo
    labels for the images, and also saving RAM by not keeping all the annotations in memory.
    :param eigenvec_dirs: list of directories containing the eigen vectors
    :param img_files: list of image files to process
    :param Ks: Ks to use for kmeans
    :param worker_dir: directory to save the temporary files
    :param tau_m: tau_m to use for votecut
    :param num_eig_vecs: number of eigen vectors to use
    :param save_period: saving period for the annotations in temp files
    :param device:
    :param resume:
    :return:
    """

    
    if len(img_files) == 0:
        print("No images left to process, exiting...")
        return
    padding = (5,5 , 5, 5)  # (left, top, right, bottom)
    Path(worker_dir).mkdir(parents=True, exist_ok=True)
    # just for tracking the skipped images
    skipped_images_file = os.path.join(worker_dir, "skipped_images.txt")
    for ind, img_file in enumerate(tqdm(img_files, desc="Creating pseudo labels")):
        try:
            print('#####')
            image_name,_ = parse_image_file(img_file)
            # load all eigen vectors for the image
            image_rgb = Image.open(img_file).convert("RGB")
            grey_current_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            # load all eigen vectors for the image
            eig_vec_groups = load_eig_vecs(eigenvec_dirs, num_eig_vecs ,image_name)
            current_boxes= get_Boxes(image_rgb=image_rgb,eig_vec_groups=eig_vec_groups, Ks=Ks, tau_m=tau_m,device=device)
            #print('current_boxes',current_boxes)
            #current_boxes= fuse_bboxes(current_boxes,[],iou_threshold=0.9)
            ref_framelist=get_adjacent_images(img_file,reference_count =2,image_type = image_type)

            ref_boxes=[]
            for ref_frame in ref_framelist:
                ref_img = Image.open(ref_frame).convert('RGB')
                ref_name,_ = parse_image_file(ref_frame)
                ref_grey_img = cv2.imread(ref_frame, cv2.IMREAD_GRAYSCALE)
                ref_eig_vec_groups = load_eig_vecs(eigenvec_dirs, num_eig_vecs ,ref_name)
                ref_box= get_Boxes(image_rgb=ref_img,eig_vec_groups=ref_eig_vec_groups, Ks=Ks, tau_m=tau_m,device=device)
                
                ref_flow = cv2.calcOpticalFlowFarneback(ref_grey_img, grey_current_img, None, pyr_scale=0.3,   
                                                                                        levels=6,       
                                                                                        winsize=25,     
                                                                                        iterations=5,    
                                                                                        poly_n=7,       
                                                                                        poly_sigma=1.5,
                                                                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                
                warped_boxes = [warp_bbox_by_flow(box, ref_flow) for box in ref_box]
                ref_boxes.append(warped_boxes)
                
            ref_boxes = [item for sublist in ref_boxes for item in sublist]
            #fused_box=filter_current_by_reference(current_boxes,ref_boxes,iou_threshold=0.7)
            #fused_box = fuse_with_reference(current_boxes,ref_boxes,iou_threshold=0.6, group_iou_threshold=0.6)
            #print(current_boxes)
            #print(ref_boxes)
            fused_box=fuse_boxes_greedy(current_boxes,ref_boxes)
            #print('fused_box#####:',fused_box)    
            #fused_box=ref_boxes
            
            crop_img=roi_align_on_image(image_rgb,fused_box)# [N, 3, 224, 224],in 0-1
            
            crops_uint8 = (crop_img * 255).byte()  # [N, 3, 224, 224]

            # 转为 processor 输入格式（PIL-like numpy）
            inputs = processor(
                images=[c.permute(1, 2, 0).cpu().numpy() for c in crops_uint8],
                return_tensors="pt"
            ).to(vit.device)
            with torch.no_grad():
                outputs = vit(**inputs)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [N, 256, 768]
            #print('patch_tokens:',patch_tokens.size())
            patch_feat_map = patch_tokens.permute(0, 2, 1).reshape(-1, 768, 16, 16).to(device)  # [N, 768, 16, 16]
            #print('patch_feat_map',patch_feat_map.size())
            with torch.no_grad():
                pred_masks,score = decoder(patch_feat_map) #[N, 1, 64, 64]
                score = torch.sigmoid(score)
                #print(pred_masks[0])
                #pred_masks = torch.sigmoid(pred_masks)        
                #print(pred_masks[0])
                # 直接二值化，阈值可调（比如 0.5）
                
                #binary_masks = (pred_masks > 0.5).float()  
                #print('score',score)
                
                mean_score = score.mean()
                threshold = min(mean_score.item(), 0.85) 
                keep_idx = score.squeeze(-1)  > threshold

                filtered_masks = pred_masks[keep_idx]
                filtered_masks = torch.sigmoid(filtered_masks)
                #print('filtered_masks:',filtered_masks.size())
                filtered_bboxes = [bbox for keep, bbox in zip(keep_idx, fused_box) if keep]
                #print('####',len(filtered_masks),len(filtered_bboxes))
                assert len(filtered_masks) == len(filtered_bboxes), \
                    f"数量不匹配: masks={len(filtered_masks)}, bboxes={len(filtered_bboxes)}"
                
                #print(len(filtered_masks))
                '''
                pred_masks_flat = filtered_masks.view(filtered_masks.size(0), -1)
                for i in range(pred_masks.size(0)):
                    mask_min = pred_masks_flat[i].min().item()
                    mask_max = pred_masks_flat[i].max().item()
                    mask_mean = pred_masks_flat[i].mean().item()
                    print(f"Mask {i}: min={mask_min:.4f}, max={mask_max:.4f}, mean={mask_mean:.4f}")
                '''
                #final_masks=map_masks_to_original_image(image_rgb,filtered_masks,filtered_bboxes)
                #print('filtered_masks',filtered_masks.size(),'filtered_bboxes',filtered_bboxes.size())
                final_masks=map_masks_to_original_image(image_rgb,filtered_masks,filtered_bboxes)
                
                #print('len(final_masks)',len(final_masks))
                polymask=mask2polygon(final_masks)
                #print('masks:',len(polymask))
            print('len(polymask)',len(polymask))
            
            if len(polymask)!= 0:
            
                write_to_xml_vit(image_file = img_file,
                            info =polymask,
                            save_folder= args.save_xml,
                            database_name = args.database)
            '''
            write_to_xml_box(image_file = img_file,
                            bboxes=fused_box,
                            save_folder= args.save_xml,
                            database_name = args.database
                           )
            '''
        except Exception as e:
            print(f"Error: {e}")
        # save the annotations to temp file for aggregation
    print('finish....')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create pseudo labels mask coco annotation file")
    parser.add_argument("--dataset-root", type=str, default="/netscratch/zlu/dataset/ytvis_2019", help="Path to coco dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Split to use")
    parser.add_argument("--Ks", type=tuple, default=(2,3), help="Ks to use for kmeans")
    #parser.add_argument("--out-file", type=str, default="/netscratch/zlu/dataset/imagenetvid_sam/cutvlervideoannotations.json", help="")
    parser.add_argument("--tau-m", type=float, default=0.2, help="")
    parser.add_argument("--models", nargs='+',
                        default=["dino_s16", "dinov2_b14", "dinov2_s14", "dino_b16", "dino_s8", "dino_b8"],
                        #default=["dino_s16", "dinov2_b14", "dinov2_s14", "dino_b16", "dino_s8", "dino_b8"],
                        help="List of models to use")
    parser.add_argument("--eig-vec-dir", type=str, default="/netscratch/zlu/dataset/ytvis_2019/cutvler/eig_vecs_train", help="Directory of images eigen vectors for each model")
    parser.add_argument("--num-eig-vecs", type=int, default=1, help="Number of eigen vectors to use")
    parser.add_argument("--save-period", type=int, default=100, help="saving period for the annotations in temp files")
    parser.add_argument("--tmp-folder", type=str, default="/netscratch/zlu/dataset/imagenetvid_sam/tmp", help="Directory to save temp files")
   # parser.add_argument("--save-tmp-files", action="store_true", help="Save temp files")
    parser.add_argument("--resume", type=bool, default=False, help="Resume from previous run")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--database", type=str, default="ytvis_2019")
    parser.add_argument("--save-xml", type=str, default="/netscratch/zlu/dataset/ytvis_2019/vitcut/train",)
    parser.add_argument("--image-type", type=str, default="jpg",choices=["jpg", "JPEG"])
     
    args = parser.parse_args()
    
    Ks = args.Ks
    tmp_folder = args.tmp_folder
    eigenvec_dirs = [f"{args.eig_vec_dir}/{model}" for model in args.models]
    #args.save_xml = args.save_xml+'/dinov2_s14/'
    if args.split == "val":
        '''
        all_image_files = []
        folder_path = args.dataset_root+'/val/'
        for filename in os.listdir(folder_path):
            if filename.startswith("ILSVRC2015_val_"):
                try:
                    file_num = int(filename.split('_')[-1])
                    if 143002 <= file_num <=177001:
                        filepath=(folder_path +'/'+filename)
                        all_image_files = all_image_files+glob(f"{filepath}/*.JPEG")
                except ValueError:
                    pass  
        '''   
        #all_image_files = glob(f"{args.dataset_root}/train/ILSVRC2015_train_00790000/*.JPEG")

        #all_image_files = glob(f"{args.dataset_root}/train/JPEGImages/0d53136c81/*.jpg")
        all_image_files = glob(f"{args.dataset_root}/train/JPEGImages/2dcc417f82/*.jpg")
        '''
        xmlfolder='/netscratch/zlu/dataset/imagenet-vid/vit/sam_xml/train/'
        exist = set(os.listdir(xmlfolder))
        imgfolder = args.dataset_root+'/train/'
        imgs=set(os.listdir(imgfolder)) 

        not_exist = imgs-exist
        all_image_files=[]
        for video in not_exist:
            #print('video',video)
        
            
            if 0<= int(video) <= 72:
                print(video)
                frames = glob(f"{args.dataset_root}/train/{video}/*.jpg")
                all_image_files=all_image_files+frames
               
        
            if video.startswith("ILSVRC2015_train_"):
                file_num = int(video.split('_')[-1]) 
                if 0<=file_num<= 1151000:
                    path=imgfolder+video
                    img= glob(f"{path}/*.JPEG")
                    all_image_files=all_image_files+img
        
        '''
        #print('val part....')
        #all_image_files = glob(f"{args.dataset_root}/val/*/*.jpg")
        
        #all_image_files =all_image_files +glob(f"{args.dataset_root}/train/1feae28e/*.jpg")
        '''
        all_image_files=[]
        path = args.dataset_root+'/val/'
        video = sorted(os.listdir(path)) 
        
        ##youtube2021 train
        ##  "0043f083b5"<= f < "30318465dc".."30318465dc"<= f < "603feaee6d"
        ##  "603feaee6d"<= f < "9349ebfd3f".."9349ebfd3f"<= f < "c01faff1ed"
        ##  "c01faff1ed"<= f <= "fffe5f8df6"
        ## val
        ## "00f88c4f0a"<= f < "4c0a64ea8f".."4c0a64ea8f"<= f < "929e8d6866"
        ##"929e8d6866"<= f < "bf730c6ae7" .."bf730c6ae7"<= f <= "ffd7c15f47"
        
        ######ovis
        ##"001ca3cb"<= f <"4023fc77".."4023fc77"<= f <"708098a9"
        ##"708098a9"<= f <"a1f4c54d" .."a1f4c54d"<= f <"d0696759"
        ## "d0696759"<= f <="ffcab778"
        #####youtubevis2019
        ###"003234408d"<= f <"374b479880" .."374b479880"<= f <"603189a03c"
        ###"603189a03c"<= f <"903e87e0d6".."903e87e0d6"<= f <"c013f42ed7"
        ### "c013f42ed7"<= f <="fffe5f8df6"
        ###youtubevis2019 valid
        ###"0062f687f1"<= f <"4294ab03bf".."4294ab03bf"<= f <"8273b59141"
        ###"8273b59141"<= f <"bf4cc89b18"   "bf4cc89b18"<= f <="ffd7c15f47"
        
        for f in video:
            if "0"<= f <= "18":
                #print(f)
                frames = glob(f"{args.dataset_root}/val/{f}/*.jpg")
                all_image_files=all_image_files+frames
        '''
    elif args.split == "train":
        '''
        main_folder = args.dataset_root+'/train/ILSVRC2015_VID_train_0002'
        exist_video = [video for video in os.listdir((f"{args.save_xml}ILSVRC2015_VID_train_0002/"))]
        all_folders = [f for f in os.listdir(main_folder)]
        unprocessed_folders = [f for f in all_folders if f not in exist_video]
        print('exist_video:',len(exist_video),'all_folders:',len(all_folders),'unprocessed_folders::',len(unprocessed_folders))
        all_image_files =[]
        for video in unprocessed_folders:
            frame = glob(f"{args.dataset_root}/train/ILSVRC2015_VID_train_0002/{video}/*.JPEG")
            all_image_files=all_image_files+frame
        '''    
            #all_image_files
        #all_image_files = glob(f"{args.dataset_root}/train/ILSVRC2015_VID_train_0000/*/*.JPEG")
        #all_image_files = glob(f"{args.dataset_root}/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00025019/*.JPEG")
        
        all_image_files = []
        folder_path = args.dataset_root+'/train/ILSVRC2015_VID_train_0003'
        for filename in os.listdir(folder_path):
            if filename.startswith("ILSVRC2015_train_"):
                try:
                    file_num = int(filename.split('_')[-1])
                    if 1124000<= file_num <= 1151000:
                        filepath=(folder_path +'/'+filename)
                        all_image_files = all_image_files+glob(f"{filepath}/*.JPEG")
                except ValueError:
                    pass  
        
        
        
        #/netscratch/zlu/dataset/imagenetvid_sam/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00677000/000790.JPEG', '/netscratch/zlu/dataset/imagenetvid_sam/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00677000/000363.JPEG
    else:
        raise ValueError(f"Invalid split {args.split} provided. Must be one of ['train', 'val']")
    print('the number of files making: ',len(all_image_files))
    
    #predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    vit = AutoModel.from_pretrained(model_name)  
    
    decoder = DINOv2SegDecoder(out_size=64).to(args.device)
    ##vit-sam2decoder_imagenet  vit_sam2decoder_youtubevis
    current_dir = Path(__file__).resolve().parent
    ckpt_path = current_dir / "student_teacher_pretrained.pt"
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    state_dict = checkpoint["student_model"]
    
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}   
    decoder.load_state_dict(state_dict)

    decoder.eval()
    
    create_ViTcut_annotations(eigenvec_dirs, all_image_files, args.Ks, tmp_folder, args.tau_m, args.num_eig_vecs, args.save_period, args.device, args.resume,vit,processor,decoder,image_type=args.image_type)
  
    #create_ViTcut_annotations(eigenvec_dirs, all_image_files, args.Ks, tmp_folder, args.tau_m, args.num_eig_vecs, args.save_period, args.device, args.resume,None,None,None,image_type=args.image_type)
    

    
    exit(0)
