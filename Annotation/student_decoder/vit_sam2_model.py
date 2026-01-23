import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import pycocotools.mask as maskUtils
from transformers import Dinov2Model, AutoImageProcessor
import torchvision.transforms.functional as TF
import logging
from datetime import datetime
import random
import PIL
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
import math
from torch.optim.lr_scheduler import _LRScheduler
from sam2.sam2_image_predictor import SAM2ImagePredictor
from concurrent.futures import ThreadPoolExecutor
import cv2
import re
import matplotlib.pyplot as plt
from torch.utils.data import Subset
# from vitCut import map_masks_to_original_image


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29507"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


class Learned2DPosEncoding(nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
        # height × width × dim
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, HW, C = x.shape
        assert HW == self.height * self.width, f"hw={HW} does not match position embedding size"
        assert C == self.dim, f"C={C} does not match position embedding dim"

        # x: [B, HW, C], HW == height * width
        return x + self.pos_embed  # broadcast add


def coco_json_to_dataset_list(json_file, images_dir=None):
    """
    Expand a COCO JSON file into a record per bbox, producing a data_list usable by a dataset.

    Args:
        json_file (str): Path to the COCO JSON file
        images_dir (str, optional): If file_name in JSON is relative, specify the image root directory here

    Returns:
        list of dict: each dict contains "img_path" and "bbox"
    """
    import json
    with open(json_file, "r") as f:
        data = json.load(f)

    # Build image_id -> image_info mapping
    image_dict = {img["id"]: img for img in data.get("images", [])}

    dataset_list = []
    for ann in data.get("annotations", []):

        image_id = ann["image_id"]
        img_info = image_dict.get(image_id)
        if img_info is None:
            continue

        file_name = img_info["file_name"]
        if images_dir is not None:
            img_path = os.path.join(images_dir, file_name)
        else:
            img_path = file_name

        dataset_list.append({
            "img_path": img_path,
            "bbox": ann["bbox"]
        })

    return dataset_list


def ts_collate_fn(batch):
    teachers = [b["teacher_img"] for b in batch]  # list, each image may have different size
    students = torch.stack([b["student"] for b in batch], dim=0)  # fixed size can be stacked
    bboxes = [b["bbox"] for b in batch]  # list, keep original bbox
    orig_img_path = [b["orig_img_path"] for b in batch]
    return {"teacher_img": teachers, "student": students, "bbox": bboxes, "orig_img_path": orig_img_path}


class TeacherStudentDataset(Dataset):
    def __init__(self, data_list, out_size=224, processor=None):
        self.data_list = data_list
        self.out_size = out_size
        self.processor = processor
        self.augment_prob = 0.5

    def __len__(self):
        return len(self.data_list)

    def crop_and_pad(self, image, bbox, out_size):
        # Crop
        x, y, w, h = map(int, bbox)
        crop = TF.crop(image, y, x, h, w)

        # Keep aspect ratio resize
        crop_w, crop_h = crop.size
        scale = out_size / max(crop_w, crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        crop_resized = crop.resize((new_w, new_h), resample=Image.BICUBIC)

        # Pad to square
        padded_image = Image.new("RGB", (out_size, out_size), (0, 0, 0))
        paste_x = (out_size - new_w) // 2
        paste_y = (out_size - new_h) // 2
        padded_image.paste(crop_resized, (paste_x, paste_y))
        return padded_image

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path, bbox = item["img_path"], item["bbox"]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Convert to RGB + PIL.Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)
        orig_w, orig_h = image.size

        do_augment = random.random() < self.augment_prob
        x, y, w, h = bbox

        # Optional data augmentation
        if do_augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                x = orig_w - x - w

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                y = orig_h - y - h

        x1, y1 = x, y
        x2, y2 = x + w, y + h
        bbox_xyxy = (x1, y1, x2, y2)

        # Student: bbox crop -> resize -> pad -> processor
        student_crop = self.crop_and_pad(image, bbox, out_size=self.out_size)
        student_tensor = self.processor(images=student_crop, return_tensors="pt")["pixel_values"][0]

        teacher_img_rgb = np.array(image)
        return {
            "teacher_img": teacher_img_rgb,        # numpy RGB
            "student": student_tensor,
            "bbox": torch.tensor(bbox_xyxy),       # aligned bbox
            "orig_img_path": img_path,
        }


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class MaskHeadConvUpsample(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=128, out_size=64):
        super().__init__()
        self.out_size = out_size
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        mask = self.mask_head(x)
        mask = F.interpolate(mask, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return mask


class DINOv2SegDecoder(nn.Module):
    def __init__(self, in_dim=768, out_size=64, num_heads=4, num_layers=6, feat_H=16, feat_W=16, num_queries=16):
        super().__init__()
        self.out_size = out_size
        self.num_queries = num_queries

        # Project backbone features
        self.encoder_conv = nn.Conv2d(in_dim, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # Learnable queries
        self.query_pos = nn.Parameter(torch.randn(num_queries, 256))

        # Positional encoding for memory
        self.pos_encoding = Learned2DPosEncoding(feat_H, feat_W, 256)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256, nhead=num_heads, dim_feedforward=512, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Mask head
        self.mask_head = MaskHeadConvUpsample(in_channels=256, hidden_channels=128, out_size=out_size)

        # Score head
        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: [B, C=768, H, W]
        B, C, H, W = x.shape

        # Backbone + projection
        memory = self.encoder_conv(x)  # [B, 256, H, W]
        memory_flat = memory.flatten(2).permute(0, 2, 1)  # [B, HW, 256]
        memory_flat = self.pos_encoding(memory_flat)

        # Expand queries to batch
        queries = self.query_pos.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, 256]

        # Transformer decoder
        decoder_out = self.transformer_decoder(tgt=queries, memory=memory_flat)  # [B, num_queries, 256]

        # Fuse decoder output and memory in parallel
        num_queries = decoder_out.shape[1]
        decoder_feat = decoder_out.unsqueeze(-1).unsqueeze(-1)  # [B, num_queries, 256, 1, 1]
        memory_exp = memory.unsqueeze(1).expand(B, num_queries, -1, H, W)  # [B, num_queries, 256, H, W]
        fused_feat = memory_exp + decoder_feat  # broadcast add [B, num_queries, 256, H, W]

        # Reshape into batch dimension for conv heads
        fused_feat = fused_feat.view(B * num_queries, 256, H, W)
        mask_out = self.mask_head(fused_feat)   # [B*num_queries, 1, outH, outW]
        score_out = self.score_head(fused_feat) # [B*num_queries, 1]

        # Reshape back to [B, num_queries, ...]
        mask_out = mask_out.view(B, num_queries, self.mask_head.out_size, self.mask_head.out_size)
        score_out = score_out.view(B, num_queries, 1)

        best_idx = score_out.squeeze(-1).argmax(dim=1)  # [B]
        best_masks = mask_out[torch.arange(B), best_idx]  # [B, outH, outW]
        best_scores = score_out[torch.arange(B), best_idx]

        return best_masks, best_scores


class WarmupCosineAnnealingRestartLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, switch_epoch=20, min_lr=1e-6, warmup_epochs=5, last_epoch=-1):
        self.total_epochs = total_epochs
        self.switch_epoch = switch_epoch
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                # Warm up linearly from 0 to base_lr
                lr = base_lr * epoch / self.warmup_epochs
            elif epoch < self.switch_epoch:
                # Cosine decay from base_lr to min_lr
                t = (epoch - self.warmup_epochs) / (self.switch_epoch - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
            else:
                # Restart: jump back to base_lr and decay again
                t = (epoch - self.switch_epoch) / (self.total_epochs - self.switch_epoch)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs


def predict_sam2_mask_score_from_crop(predictor, crop_img_tensor, device, output_size=None):
    """
    predictor: SAM2ImagePredictor
    crop_img_tensor: Tensor [C,H,W], float in [0,1], cropped image
    device: torch.device
    output_size: tuple or None, resize output mask size

    Returns:
        mask_tensor: [1,H,W] float tensor aligned with crop
        score: scalar tensor
    """
    crop_img_np = crop_img_tensor.permute(1, 2, 0).cpu().numpy()  # HWC
    crop_img_np = (crop_img_np * 255).astype(np.uint8)  # uint8

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            predictor.set_image(crop_img_np)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,  # crop is the full input, no box needed
                multimask_output=False,
            )
            mask_tensor = torch.from_numpy(masks[0]).unsqueeze(0).float().to(device)

    if output_size is not None:
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False
        ).squeeze(0)

    mask_tensor = (mask_tensor > 0.5).float()
    score_tensor = torch.clamp(torch.tensor(scores[0], device=device), 0.0, 1.0)

    return mask_tensor, score_tensor


def visualize_teacher_student(orig_img, bbox, student_crop, teacher_mask, pred_mask, save_dir, prefix="sample"):
    """
    orig_img: HWC numpy (RGB), original image used as teacher input
    bbox: tensor/list [x1,y1,x2,y2], teacher bbox
    student_crop: [C,H,W] tensor (student input)
    teacher_mask: [1,H,W] teacher mask (already aligned to student size)
    pred_mask: [1,H,W] student output logits/probabilities
    """
    os.makedirs(save_dir, exist_ok=True)

    # Teacher crop from original + bbox
    x1, y1, x2, y2 = map(int, bbox)
    teacher_crop = orig_img[y1:y2, x1:x2, :]

    # Student crop (tensor -> numpy)
    student_img = student_crop.permute(1, 2, 0).cpu().numpy()
    student_img = np.clip(student_img, 0, 1)

    # Masks to numpy
    teacher_mask = teacher_mask.squeeze(0).cpu().numpy()
    pred_mask = torch.sigmoid(pred_mask).squeeze(0).detach().cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(teacher_crop)
    axs[0, 0].set_title("Teacher Crop")
    axs[0, 1].imshow(teacher_crop)
    axs[0, 1].imshow(teacher_mask, alpha=0.4, cmap="Reds")
    axs[0, 1].set_title("Teacher Mask on Crop")

    axs[1, 0].imshow(student_img)
    axs[1, 0].set_title("Student Crop")
    axs[1, 1].imshow(student_img)
    axs[1, 1].imshow(pred_mask > 0.5, alpha=0.4, cmap="Blues")
    axs[1, 1].set_title("Student Pred Mask")

    for ax in axs.flat:
        ax.axis("off")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_path}")


def predict_sam2_mask_score(predictor, img_rgb, bbox, device, student_input_size=None, decoder_output_size=None):
    """
    Inputs:
        predictor: SAM2ImagePredictor
        img_rgb: numpy RGB image (H,W,3)
        bbox: tensor[4] (x1,y1,x2,y2)
        device: torch.device
        decoder_output_size: (H,W) or None, whether to resize predicted mask

    Returns:
        mask: tensor[1, H, W] predicted mask
        score: scalar tensor
    """
    bbox_int = bbox.cpu().numpy().astype(int).tolist()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            predictor.set_image(img_rgb)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_int,
                multimask_output=False,
            )
            best_idx = int(torch.argmax(torch.tensor(scores)))
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            mask_tensor = torch.from_numpy(best_mask).unsqueeze(0).float().to(device)

    H_img, W_img, _ = img_rgb.shape
    croped_mask = align_teacher_mask_to_orig(
        mask_tensor, (H_img, W_img), bbox, decoder_output_size=decoder_output_size
    )
    score_tensor = torch.clamp(torch.tensor(best_score, device=device), 0.0, 1.0)

    return croped_mask, score_tensor


def align_teacher_mask_to_orig(mask_tensor, orig_img_size, crop_bbox, decoder_output_size=(128, 128)):
    """
    Map teacher mask back to original image coordinates, and optionally resize to decoder output size.

    mask_tensor: [1, H_mask, W_mask] mask in original image space
    orig_img_size: (H_img, W_img)
    crop_bbox: [x1, y1, x2, y2]
    decoder_output_size: (H_out, W_out) or None
    """
    H_img, W_img = orig_img_size
    x1, y1, x2, y2 = map(int, crop_bbox.tolist())
    _, H, W = mask_tensor.shape

    # Crop mask to bbox; pad zeros if bbox exceeds image boundary
    crop_h, crop_w = max(y2 - y1, 1), max(x2 - x1, 1)
    mask_crop = torch.zeros((1, crop_h, crop_w), device=mask_tensor.device)

    x1_c = max(x1, 0)
    y1_c = max(y1, 0)
    x2_c = min(x2, W)
    y2_c = min(y2, H)
    if x2_c > x1_c and y2_c > y1_c:
        mask_crop[:, (y1_c - y1):(y2_c - y1), (x1_c - x1):(x2_c - x1)] = mask_tensor[:, y1_c:y2_c, x1_c:x2_c]

    # Optionally resize to decoder output size
    if decoder_output_size is not None:
        mask_orig = F.interpolate(
            mask_crop.unsqueeze(0), size=decoder_output_size, mode='bilinear', align_corners=False
        ).squeeze(0)
    else:
        mask_orig = mask_crop

    return mask_orig


def align_teacher_mask_to_crop(mask_tensor, crop_bbox, student_input_size=(224, 224), decoder_output_size=(64, 64)):
    """
    mask_tensor: [1,H,W] mask in original image space
    crop_bbox: [x1, y1, x2, y2]
    student_input_size: (H_in, W_in)
    decoder_output_size: (H_out, W_out)
    """
    x1, y1, x2, y2 = map(int, crop_bbox.tolist())
    _, H, W = mask_tensor.shape
    H_in, W_in = student_input_size

    # Crop mask to bbox; pad zeros if bbox exceeds image boundary
    crop_h, crop_w = max(y2 - y1, 1), max(x2 - x1, 1)
    mask_crop = torch.zeros((1, crop_h, crop_w), device=mask_tensor.device)
    x1_c = max(x1, 0)
    y1_c = max(y1, 0)
    x2_c = min(x2, W)
    y2_c = min(y2, H)
    if x2_c > x1_c and y2_c > y1_c:
        mask_crop[:, (y1_c - y1):(y2_c - y1), (x1_c - x1):(x2_c - x1)] = mask_tensor[:, y1_c:y2_c, x1_c:x2_c]

    # Aspect-ratio resize + padding to student input size
    h_crop, w_crop = mask_crop.shape[1:]
    scale = max(H_in / h_crop, W_in / w_crop)
    new_h = max(int(h_crop * scale), 1)
    new_w = max(int(w_crop * scale), 1)
    mask_resized = F.interpolate(mask_crop.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)

    pad_h = H_in - new_h
    pad_w = W_in - new_w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    mask_aligned_input = F.pad(mask_resized, (pad_left, pad_right, pad_top, pad_bottom))

    # Resize to decoder output size
    if decoder_output_size is not None:
        mask_tensor = F.interpolate(mask_aligned_input, size=decoder_output_size, mode='bilinear', align_corners=False).squeeze(0)

    return mask_tensor


def visualize_teacher_student_mask_alignment(orig_img_path, bbox, teacher_mask, student_mask, save_path):
    import cv2, os, torch, matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F

    # Read original image
    img_bgr = cv2.imread(orig_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    x1, y1, x2, y2 = map(int, bbox.tolist())
    crop_h, crop_w = y2 - y1, x2 - x1

    # Resize masks to bbox size (nearest for crisp boundaries)
    teacher_mask_resized = F.interpolate(
        teacher_mask.unsqueeze(0).unsqueeze(0),
        size=(crop_h, crop_w),
        mode='nearest'
    ).squeeze().cpu().numpy()

    student_mask_resized = F.interpolate(
        student_mask.unsqueeze(0).unsqueeze(0),
        size=(crop_h, crop_w),
        mode='nearest'
    ).squeeze().cpu().detach().numpy()

    # Teacher overlay
    teacher_img = img_rgb.copy()
    teacher_region = teacher_img[y1:y2, x1:x2]
    teacher_mask_bool = teacher_mask_resized > 0
    teacher_region[teacher_mask_bool, 0] = 1.0  # red
    teacher_region[teacher_mask_bool, 1] = 0.0
    teacher_region[teacher_mask_bool, 2] = 0.0
    # Draw bbox (green)
    cv2.rectangle(teacher_img, (x1, y1), (x2, y2), color=(0, 1, 0), thickness=10)

    # Student overlay
    student_img = img_rgb.copy()
    student_region = student_img[y1:y2, x1:x2]
    student_mask_bool = student_mask_resized > 0
    student_region[student_mask_bool, 0] = 0.0
    student_region[student_mask_bool, 1] = 0.0
    student_region[student_mask_bool, 2] = 1.0  # blue
    # Draw bbox (green)
    cv2.rectangle(student_img, (x1, y1), (x2, y2), color=(0, 1, 0), thickness=10)

    # Concatenate left-right
    combined_img = np.concatenate([teacher_img, student_img], axis=1)
    combined_bgr = (combined_img * 255).astype(np.uint8)
    combined_bgr = cv2.cvtColor(combined_bgr, cv2.COLOR_RGB2BGR)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined_bgr)


def load_student_weights(model, weight_path, device='cpu'):
    checkpoint = torch.load(weight_path, map_location=device)

    if 'student_model' not in checkpoint:
        raise KeyError("No 'student_model' key found in the checkpoint.")

    state_dict = checkpoint['student_model']

    # Current model state dict
    model_dict = model.state_dict()

    # Keep only matching weights
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Update model parameters
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Report layers not loaded
    not_loaded = [k for k in model_dict if k not in pretrained_dict]
    if not_loaded:
        print("No weights loaded for:", not_loaded)
    else:
        print("All weights loaded.")

    model.to(device)
    return model


def map_student_masks_to_orig_batch(image_list, pred_masks, fused_boxes, out_size=None):
    N = pred_masks.shape[0]
    assert len(image_list) == N, f"len(image_list)={len(image_list)} and pred_masks batch={N} do not match"

    aligned_masks = []
    for i in range(N):
        # Read original image
        img_bgr = cv2.imread(image_list[i], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_list[i]}")
        orig_h, orig_w = img_bgr.shape[:2]

        # Current mask
        mask = pred_masks[i]  # [1, Hm, Wm]

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # -> [1,1,Hm,Wm]
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)  # -> [1,1,Hm,Wm]

        # Current bbox
        box = fused_boxes[i]
        if isinstance(box, torch.Tensor):
            box = box.detach().cpu().numpy().tolist()
        elif isinstance(box, np.ndarray):
            box = box.tolist()
        x1, y1, x2, y2 = map(int, box)

        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)

        # Resize mask to bbox size
        stu_mask_resized = F.interpolate(
            mask, size=(box_h, box_w), mode='bilinear', align_corners=False
        ).squeeze(0)  # [1, h, w]

        # Then resize to out_size
        mask_final = F.interpolate(
            stu_mask_resized.unsqueeze(0), size=out_size, mode='bilinear', align_corners=False
        )  # [1,1,H_out,W_out]

        aligned_masks.append(mask_final)

    if aligned_masks:
        aligned_masks = torch.cat(aligned_masks, dim=0)  # [N,1,H_out,W_out]
    else:
        aligned_masks = torch.empty(0, 1, *out_size, device=student_masks.device)

    return aligned_masks


def boundary_loss(pred_logits, target_mask):
    """
    pred_logits: [B,1,H,W] logits
    target_mask: [B,1,H,W] soft mask in [0,1]
    """
    pred_probs = torch.sigmoid(pred_logits)

    # Sobel kernels
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=pred_probs.dtype, device=pred_probs.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    # Edge extraction
    gx_pred = F.conv2d(pred_probs, sobel_x, padding=1)
    gy_pred = F.conv2d(pred_probs, sobel_y, padding=1)
    gx_target = F.conv2d(target_mask, sobel_x, padding=1)
    gy_target = F.conv2d(target_mask, sobel_y, padding=1)

    # L1 loss to align boundaries
    loss = F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)
    return loss


def dice_loss_with_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def train_ddp(rank, world_size):
    print('Entering setup...')
    total_epoch = 40
    batch_size = 64
    do_transform = False
    img_dir = '/netscratch/zlu/dataset/youtubevos/train/JPEGImages/'
    datasetfile = '/netscratch/zlu/dataset/youtubevos/bbox/ytvis_origin_train_bbox.json'
    pretrained_model = 'facebook/dinov2-base'

    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    set_seed(42 + rank)
    print('Entering training...')

    if rank == 0:
        log_file = f"/netscratch/zlu/CutvLER/output/youtube_vit_2/train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    init_checkpoint_path = "/netscratch/zlu/CutvLER/output/vit_sam2decoder_youtubevis/student_teacher_epoch24.pt"
    checkpoint_path = None
    start_epoch = 0  # continue from a specific epoch if needed

    student_backbone = Dinov2Model.from_pretrained(pretrained_model).to(device)
    student_backbone.eval()
    teacher_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    student_model = DINOv2SegDecoder().to(device)
    student_model.apply(init_weights)

    # Initialize query_pos with small values (instead of default randn)
    nn.init.trunc_normal_(student_model.query_pos, std=0.02)

    student_model = nn.parallel.DistributedDataParallel(student_model, device_ids=[rank])

    proc = AutoImageProcessor.from_pretrained(pretrained_model)

    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_score = nn.BCELoss()

    if init_checkpoint_path is not None and os.path.exists(init_checkpoint_path):
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        init_checkpoint = torch.load(init_checkpoint_path, map_location=map_location)
        student_model.module.load_state_dict(init_checkpoint['student_model'])
        logger.warning(f"Loaded init checkpoint from {init_checkpoint_path}")

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        match = re.search(r'epoch(\d+)', checkpoint_path)
        if match:
            start_epoch = int(match.group(1))

        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        student_model.module.load_state_dict(checkpoint['student_model'])

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.last_epoch = start_epoch - 1
            logger.warning("Optimizer and scheduler state loaded.")

        logger.warning(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, training from scratch.")
        start_epoch = 0

    data_list = coco_json_to_dataset_list(datasetfile, images_dir=img_dir)
    train_dataset = TeacherStudentDataset(
        data_list=data_list,
        out_size=224,
        processor=proc
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=ts_collate_fn
    )

    optimizer = torch.optim.AdamW(
        list(student_model.parameters()), lr=2e-4, weight_decay=1e-2
    )
    scheduler = WarmupCosineAnnealingRestartLR(
        optimizer,
        total_epochs=total_epoch,
        switch_epoch=20,
        min_lr=1e-6,
        warmup_epochs=5
    )
    print('Entering epochs...')

    for epoch in range(start_epoch, total_epoch):
        logger.warning(f"Epoch {epoch} lr: {scheduler.get_last_lr()[0]}")

        train_sampler.set_epoch(epoch)
        student_model.train()

        z = 0

        for batch in train_loader:
            teacher_imgs = batch["teacher_img"]            # numpy RGB list
            student_tensors = batch["student"].to(device)  # [B, C, H, W]
            bboxes = batch["bbox"]
            orig_img_paths = batch['orig_img_path']

            teacher_masks = []
            teacher_scores = []

            for i in range(len(teacher_imgs)):
                mask, score = predict_sam2_mask_score(
                    predictor=teacher_predictor,
                    img_rgb=teacher_imgs[i],
                    bbox=bboxes[i],
                    device=device,
                    decoder_output_size=(64, 64)
                )

                mask = mask.float().to(device)  # [1,H,W]
                teacher_masks.append(mask)
                teacher_scores.append(score)

            teacher_masks = torch.stack(teacher_masks)   # [B, 1, 64, 64]
            teacher_scores = torch.stack(teacher_scores) # [B]
            teacher_scores = torch.clamp(teacher_scores, 0.0, 1.0)

            selected_indices = (teacher_scores > 0.7).nonzero(as_tuple=True)[0]
            if len(selected_indices) == 0:
                continue

            with torch.no_grad():
                student_feats = student_backbone(pixel_values=student_tensors).last_hidden_state
            student_feats = student_feats[:, 1:, :]
            B, N, C = student_feats.shape
            H = W = int(N ** 0.5)
            student_feats = student_feats.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

            pred_masks, pred_scores = student_model(student_feats)
            pred_masks = pred_masks.unsqueeze(1)
            pred_scores = torch.sigmoid(pred_scores).squeeze(-1)

            selected_pred_scores = pred_scores[selected_indices]
            selected_teacher_scores = teacher_scores[selected_indices]
            selected_masks = teacher_masks[selected_indices]
            selected_pred_masks = pred_masks[selected_indices]

            loss_score = criterion_score(selected_pred_scores, selected_teacher_scores)

            selected_masks = selected_masks.clamp(0, 1)

            bce_los = criterion_seg(selected_pred_masks, selected_masks)
            dice_loss = dice_loss_with_logits(selected_pred_masks, selected_masks)
            b_loss = boundary_loss(selected_pred_masks, selected_masks)
            loss_seg = 0.5 * bce_los + 0.3 * dice_loss + 0.2 * b_loss
            print('bce_los', bce_los, 'dice_loss', dice_loss, 'b_loss', b_loss, 'all:', loss_seg)

            loss = loss_seg + loss_score

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if rank == 0:
            print(f"[Epoch {epoch+1:02d}] Total: {loss.item():.4f} | SegLoss: {loss_seg.item():.4f} | ScoreLoss: {loss_score.item():.4f}")
            logger.warning(f"[Epoch {epoch+1:02d}] Total: {loss.item():.4f} | SegLoss: {loss_seg.item():.4f} | ScoreLoss: {loss_score.item():.4f}")

            if (epoch + 1) % 1 == 0:
                torch.save({
                    'student_model': student_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, f'/netscratch/zlu/CutvLER/output/youtube_vit_2/student_teacher_epoch{epoch+1}.pt')
                logger.warning(f"Saved checkpoint at epoch {epoch+1}")

    cleanup_ddp()


class SegCropDataset(Dataset):
    def __init__(self, pt_dir, augment=True, keep_orig_prob=0.4):
        self.pt_files = sorted([
            os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")
        ])
        self.augment = augment
        self.keep_orig_prob = keep_orig_prob  # e.g., keep original image with some probability

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        image = data['image']            # bbox-cropped image tensor
        mask = data['mask']              # bbox mask tensor
        bbox = data['bbox']              # bbox coords, e.g. [x1, y1, x2, y2]
        orig_img_path = data['orig_img_path']  # saved when building .pt

        return {
            'bbox_crop': image,
            'mask': mask,
            'bbox': bbox.clone().detach().float(),
            'orig_img_path': orig_img_path
        }


@torch.no_grad()
def evaluate(student_model, val_loader, backbone, device):
    student_model.eval()
    total_iou = 0.0
    total = 0
    all_scores = []
    all_labels = []
    print('Entering validation batches...')
    for batch in val_loader:
        bbox_crops = batch['bbox_crop'].to(device)
        masks = batch['mask'].to(device).float()

        feats = backbone(pixel_values=bbox_crops).last_hidden_state
        feats = feats[:, 1:, :]
        B, N, C = feats.shape
        H = W = int(N ** 0.5)
        feats = feats.permute(0, 2, 1).reshape(B, C, H, W)

        pred_masks, pred_scores = student_model(feats)
        pred_scores = torch.sigmoid(pred_scores)
        pred_bin = (torch.sigmoid(pred_masks) > 0.2).float()
        pred_bin = pred_bin.unsqueeze(1)
        pred_bin = F.interpolate(pred_bin, size=masks.shape[2:], mode='bilinear', align_corners=False)

        masks = (masks > 0.2).float()

        intersection = (pred_bin * masks).sum(dim=(1, 2, 3))
        union = ((pred_bin + masks) > 0).sum(dim=(1, 2, 3))
        iou = (intersection / union.clamp(min=1e-6))

        total_iou += iou.sum().item()
        total += B

        # Label: whether mask is empty (object present or not)
        has_object = (masks.view(B, -1).sum(dim=1) > 0).float()
        scores_prob = torch.sigmoid(pred_scores).squeeze(-1)
        all_scores.append(scores_prob.cpu())
        all_labels.append(has_object.cpu())

    # Multi-GPU synchronization
    total_iou_tensor = torch.tensor(total_iou, device=device)
    total_tensor = torch.tensor(total, device=device)
    dist.all_reduce(total_iou_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    if dist.get_rank() == 0:
        final_iou = total_iou_tensor.item() / total_tensor.item()
        pred_has_obj = (all_scores > 0.5).float()
        score_acc = (pred_has_obj == all_labels).float().mean().item()
        logger.info(f"[Eval] mIoU: {final_iou:.4f}, Score Accuracy: {score_acc:.4f}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    set_seed(42 + rank)

    if rank == 0:
        log_file = f"/netscratch/zlu/CutvLER/output/youtube_vit_2/val_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    global logger
    logger = logging.getLogger()
    pretrained_model = 'facebook/dinov2-base'

    backbone = Dinov2Model.from_pretrained(pretrained_model).to(device)
    backbone.eval()
    print('Step 1')

    student_model = DINOv2SegDecoder().to(device)
    checkpoint = torch.load(
        "/netscratch/zlu/CutvLER/output/youtube_vit_2/student_teacher_epoch18.pt",
        map_location={"cuda:0": f"cuda:{rank}"}
    )
    student_model.load_state_dict(checkpoint['student_model'])
    student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[rank])
    student_model.eval()
    print('Step 3')

    val_dataset = SegCropDataset("/netscratch/zlu/dataset/youtubevos/sam2/pt_val/", augment=False)

    print('Loading validation data...')
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler, num_workers=2, pin_memory=True)

    evaluate(student_model, val_loader, backbone, device)

    cleanup_ddp()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_ddp(rank, world_size)
    #eval_ddp(rank, world_size)


def get_student_model_param_count(checkpoint_path):
    """
    Compute the parameter count of student_model stored in a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file (.pt or .pth)

    Returns:
        total_params (int): total number of parameters in student_model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda')

    if 'student_model' not in checkpoint:
        raise ValueError("No 'student_model' key found in the checkpoint.")

    state_dict = checkpoint['student_model']
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"Student model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    return total_params


class StudentFullModel(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        # backbone output last_hidden_state [B, N, C]
        feats = self.backbone(x)
        # decoder forward returns (mask, score)
        masks, scores = self.decoder(feats)
        return masks


def compute_student_gflops(backbone_model, decoder_model, input_shape=(1, 3, 224, 224), device='cuda'):
    """
    Compute GFLOPs of student backbone + decoder.
    """
    from ptflops import get_model_complexity_info

    full_model = StudentFullModel(backbone_model, decoder_model).to(device)
    full_model.eval()

    macs, params = get_model_complexity_info(
        full_model,
        input_res=input_shape[1:],  # (C,H,W)
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )

    gflops = macs / 1e9 * 2  # FLOPs = 2 * MACs
    print(f"Student Full Model GFLOPs: {gflops:.2f} GFLOPs")
    print(f"Student Full Model Parameters: {params / 1e6:.2f} M")
    return gflops, params


def get_resnet50_backbone():
    resnet = models.resnet50(pretrained=True)
    # Remove fc layer, keep convolutional features only
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    return backbone


class SwinBackboneWrapper(nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.swin = swin_model

    def forward(self, x):
        feats = self.swin(x)[-1]  # [B, H, W, C]
        # Convert to [B, C, H, W] for Conv2d
        feats = feats.permute(0, 3, 1, 2).contiguous()
        return feats


if __name__ == "__main__":
    main()
    