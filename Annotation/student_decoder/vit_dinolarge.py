
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
from PIL import Image
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


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  
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
        assert HW == self.height * self.width, f"hw={HW}not fit to position embedding size"
        assert C == self.dim, f"C={C}not fit to position embedding"
        
        # x: [B, HW, C], HW == height * width
        return x + self.pos_embed  # 广播加


class SegCropDataset(Dataset):
    def __init__(self, pt_dir, augment=True, keep_orig_prob=0.4):
        self.pt_files = sorted([
            os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")
        ])
        self.augment = augment
        self.keep_orig_prob = keep_orig_prob  # 比如 30% 保留原图

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        image = data['image']     # bbox crop图像 tensor
        mask = data['mask']       # bbox mask tensor
        bbox = data['bbox']       # bbox坐标，例如 [x1, y1, x2, y2]
        orig_img_path = data['orig_img_path']  # 你需要在制作.pt时保存这个字段

        # 数据增强
        if self.augment and random.random() > self.keep_orig_prob:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.8:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.7:
                image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
        #print('image++++++++++,',image.size())
        return {
            'bbox_crop': image,
            'mask': mask,
            'bbox':  bbox.clone().detach().float(),
            'orig_img_path': orig_img_path
        }


class TeacherClassifier(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # 输出概率



class DINOv2SegDecoderWithScore(nn.Module): 
    def __init__(self, in_dim=1024, out_size=56, num_heads=4, num_layers=6, feat_H=16, feat_W=16):
        super().__init__()
        self.out_size = out_size
        self.encoder_conv = nn.Conv2d(in_dim, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.pos_encoding = Learned2DPosEncoding(feat_H, feat_W, 256)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=256, nhead=num_heads, dim_feedforward=512, batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, 2, stride=2),  # mask output
        )

        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # no sigmoid
        )


    def forward(self, x):  # x: [B, C=768, H, W]
        x = self.encoder_conv(x)  # [B, 256, H, W]
        B, C, H, W = x.shape

        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        x_flat = self.pos_encoding(x_flat)     # 加上位置编码

        for layer in self.transformer_layers:
            x_flat = layer(x_flat)

        #x = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        x = x_flat.permute(0, 2, 1).contiguous().reshape(B, C, H, W) #DDP
        x = self.relu(x)

        mask_out = self.seg_decoder(x).contiguous()
        mask_out = F.interpolate(mask_out, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        score_out = self.score_head(x)
        return mask_out, score_out



def crop_teacher_feat(teacher_feats, bboxes, orig_img_sizes, patch_size=14):
    """
    从 teacher_feats 中提取每张图中 bbox 对应区域的特征表示。

    Args:
        teacher_feats: [B, N+1, C]，来自 ViT 的输出（包含 CLS token）
        bboxes: Tensor[B, 4]，格式为 [x1, y1, x2, y2]，以原图尺寸为单位
        orig_img_sizes: List[Tuple[H, W]]，每张图原始大小
        patch_size: ViT 的 patch 大小，DINOv2 是 14

    Returns:
        Tensor[B, C]，每个 bbox 的平均池化特征
    """
    B, N_plus_1, C = teacher_feats.shape
    H_feat = W_feat = int((N_plus_1 - 1) ** 0.5)
    assert H_feat * W_feat == N_plus_1 - 1, f"Cannot reshape to square: N={N_plus_1 - 1}"

    patch_tokens = teacher_feats[:, 1:, :]
    feat_maps = patch_tokens.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)

    results = []

    for i in range(B):
        feat = feat_maps[i]  # [C, H_feat, W_feat]
        x1, y1, x2, y2 = bboxes[i]  # 原图尺寸
        H_img, W_img = orig_img_sizes[i]

        # 将 bbox 映射到 feature map 上
        scale_x = W_feat / W_img
        scale_y = H_feat / H_img

        fx1 = int(x1 * scale_x)
        fy1 = int(y1 * scale_y)
        fx2 = int(x2 * scale_x)
        fy2 = int(y2 * scale_y)

        fx1 = max(fx1, 0)
        fy1 = max(fy1, 0)
        fx2 = min(fx2, W_feat - 1)
        fy2 = min(fy2, H_feat - 1)

        if fx2 < fx1 or fy2 < fy1:
            pooled = torch.zeros(C, device=teacher_feats.device)
        else:
            region = feat[:, fy1:fy2+1, fx1:fx2+1]  # [C, h, w]
            pooled = region.mean(dim=(1, 2))  # [C]

        results.append(pooled)

    return torch.stack(results, dim=0)  # [B, C]
    

def check_tensor_grad(tensor, name="Tensor"):
    print(f"{name}: requires_grad = {tensor.requires_grad}, grad_fn = {tensor.grad_fn}")


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, min_lr=1e-6, warmup_epochs=5, last_epoch=-1):
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                # Linear warmup
                lr = base_lr * epoch / self.warmup_epochs
            else:
                # Cosine annealing
                t = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs



def train_ddp(rank, world_size):
    print('into setup')
    total_epoch = 40
    initial_percent = 0.2  # 起始10%
    max_percent = 0.5    
    pretrained_model='facebook/dinov2-large'
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    set_seed(42 + rank)
    print('into training.....')
    if rank == 0:
        log_file = f"/netscratch/zlu/CutvLER/output/vit_dinolarge2/train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    teacher_backbone = Dinov2Model.from_pretrained(pretrained_model).to(device)
    teacher_backbone.eval()
    for param in teacher_backbone.parameters():
        param.requires_grad = False
    
    teacher_classifier = TeacherClassifier(feat_dim=1024).to(device)
    teacher_classifier.eval()
    for param in teacher_classifier.parameters():
        param.requires_grad = False
    
    student_model = DINOv2SegDecoderWithScore().to(device)

    #teacher_classifier = nn.parallel.DistributedDataParallel(teacher_classifier, device_ids=[rank])
    student_model = nn.parallel.DistributedDataParallel(student_model, device_ids=[rank])

    proc = AutoImageProcessor.from_pretrained(pretrained_model)
    optimizer = torch.optim.AdamW(
     list(student_model.parameters()), lr=1e-4
        )
    #scheduler = CosineAnnealingLR(optimizer, T_max=total_epoch)
    scheduler = WarmupCosineAnnealingLR(
                optimizer,
                total_epochs=total_epoch,
                min_lr=1e-6,
                warmup_epochs=5
            )


    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_score = nn.BCELoss()
    
    
    train_dataset = SegCropDataset("/netscratch/zlu/dataset/youtubevos/vit/dinov2-large/pt_train/", augment=True)
    #val_dataset = SegCropDataset("/netscratch/zlu/dataset/youtubevos/vit/dinov2base/pt_val/", augment=False)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    #val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=2, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=128, sampler=val_sampler)
    #####read checkpoint#########
    checkpoint_path ="/netscratch/zlu/CutvLER/output/vit_dinolarge2/student_teacher_epoch38.pt"
    start_epoch = 38  # 从第5轮继续

    if checkpoint_path!=None and os.path.exists(checkpoint_path):
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        student_model.module.load_state_dict(checkpoint['student_model'])
        #teacher_classifier.module.load_state_dict(checkpoint['teacher_classifier'])
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.last_epoch = start_epoch - 1
            logger.info(f"Optimizer and scheduler state loaded.")

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, training from scratch.")
        start_epoch = 0

    print('into epoch...')
    for epoch in range(start_epoch, total_epoch):
        logger.info(f"Epoch {epoch} lr: {scheduler.get_last_lr()[0]}")
        #top_k_percent = initial_percent + (max_percent - initial_percent) * epoch / (total_epoch - 1)
        #print('top_k_percent',top_k_percent)
        train_sampler.set_epoch(epoch)
        student_model.train()
        #teacher_classifier.train()
        
        for batch in train_loader:
            bbox_crops = batch['bbox_crop'].to(device)
            masks = batch['mask'].to(device).float()
            bboxes = batch['bbox']
            orig_img_paths = batch['orig_img_path']

            orig_imgs = []
            orig_img_sizes = []
            for p in orig_img_paths:
                img = PIL.Image.open(p).convert("RGB")
                orig_imgs.append(img)
                orig_img_sizes.append(img.size[::-1])

            inputs = proc(images=orig_imgs, return_tensors="pt").to(device)
            
            with torch.no_grad():
                teacher_feats = teacher_backbone(pixel_values=inputs.pixel_values).last_hidden_state
                teacher_feat_vecs = crop_teacher_feat(teacher_feats, bboxes, orig_img_sizes)
                teacher_scores = teacher_classifier(teacher_feat_vecs).squeeze(-1)
            selected_indices = (teacher_scores > 0.5).nonzero(as_tuple=True)[0]
            if len(selected_indices) == 0:
                continue
            #check_tensor_grad(teacher_scores, "teacher_scores")
            if (torch.isnan(teacher_scores).any()or torch.isinf(teacher_scores).any()):
                logger.warning(f"Abnormal teacher_scores detected, skipping batch.")
                continue 
            #print('teacher.....',teacher_scores)
            with torch.no_grad():
                student_feats = teacher_backbone(pixel_values=bbox_crops).last_hidden_state
            student_feats = student_feats[:, 1:, :]
            B, N, C = student_feats.shape
            H = W = int(N ** 0.5)
            student_feats = student_feats.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            student_feats = F.normalize(student_feats, dim=1)

            pred_masks, pred_scores = student_model(student_feats)
            #pred_scores = pred_scores.squeeze(-1)
            #print('predict....',pred_scores)
            pred_scores = torch.sigmoid(pred_scores).squeeze(-1)  
           
            #k = max(1, int(B * top_k_percent))
            #print('k',k)
            #topk_indices = torch.topk(teacher_scores, k).indices
            selected_pred_scores = pred_scores[selected_indices]
            selected_teacher_scores = teacher_scores[selected_indices]
            selected_masks = masks[selected_indices]
            selected_pred_masks = pred_masks[selected_indices]
            #print('student:',selected_pred_scores,'teacher:',selected_teacher_scores)

            loss_score = criterion_score(selected_pred_scores, selected_teacher_scores)
            #logger.info(f"loss_score:: {loss_score}")
            masks_resized = F.interpolate(selected_masks, size=selected_pred_masks.shape[2:], mode="bilinear", align_corners=False)
            loss_seg = criterion_seg(selected_pred_masks, masks_resized)
            loss = loss_seg + loss_score
            #print('loss:',loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if rank == 0:
            print(f"[Epoch {epoch+1:02d}] Total: {loss.item():.4f} | SegLoss: {loss_seg.item():.4f} | ScoreLoss: {loss_score.item():.4f}")
            logger.info(f"[Epoch {epoch+1:02d}] Total: {loss.item():.4f} | SegLoss: {loss_seg.item():.4f} | ScoreLoss: {loss_score.item():.4f}")

            if (epoch + 1) % 1 == 0:
                torch.save({
                    'student_model': student_model.module.state_dict(),
                    #'teacher_classifier': teacher_classifier.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, f'/netscratch/zlu/CutvLER/output/vit_dinolarge2/student_teacher_epoch{epoch+1}.pt')
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
                #evaluate(student_model, val_loader, teacher_backbone, device)
    
    cleanup_ddp()

@torch.no_grad()
def evaluate(student_model, val_loader, teacher_backbone, device):
    student_model.eval()
    total_iou = 0.0
    total = 0
    all_scores = []
    all_labels = []

    for batch in val_loader:
        bbox_crops = batch['bbox_crop'].to(device)
        masks = batch['mask'].to(device).float()

        feats = teacher_backbone(pixel_values=bbox_crops).last_hidden_state
        feats = feats[:, 1:, :]
        B, N, C = feats.shape
        H = W = int(N ** 0.5)
        feats = feats.permute(0, 2, 1).reshape(B, C, H, W)

        pred_masks, pred_scores = student_model(feats)
        pred_bin = (torch.sigmoid(pred_masks) > 0.5).float()
        masks = F.interpolate(masks, size=pred_masks.shape[2:], mode="bilinear", align_corners=False)

        intersection = (pred_bin * masks).sum(dim=(1,2,3))
        union = ((pred_bin + masks) > 0).sum(dim=(1,2,3))
        iou = (intersection / union.clamp(min=1e-6))

        total_iou += iou.sum().item()
        total += B

        # 根据mask是否为空判断label（是否含物体）
        has_object = (masks.view(B, -1).sum(dim=1) > 0).float()
        print("has_object labels:", has_object[:10].cpu().numpy())
        print("sigmoid(pred_scores):", torch.sigmoid(pred_scores[:10]).cpu().numpy())
        scores_prob = torch.sigmoid(pred_scores).squeeze(-1)
        all_scores.append(scores_prob.cpu())
        all_labels.append(has_object.cpu())

    # 多卡同步
    total_iou_tensor = torch.tensor(total_iou, device=device)
    total_tensor = torch.tensor(total, device=device)
    dist.all_reduce(total_iou_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    if dist.get_rank() == 0:
        final_iou = total_iou_tensor.item() / total_tensor.item()
        # 计算score的简单准确率（阈值0.5）
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
    print('into eval...')
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    set_seed(42 + rank)
    if rank == 0:
        log_file = f"/netscratch/zlu/CutvLER/output/vit_dinolarge2/val_log_13.log"
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
    student_model = DINOv2SegDecoderWithScore().to(device)
    teacher_backbone = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device)
    teacher_backbone.eval()

    checkpoint = torch.load("/netscratch/zlu/CutvLER/output/vit_dinolarge2/student_teacher_epoch13.pt", 
                            map_location={"cuda:0": f"cuda:{rank}"})
    student_model.load_state_dict(checkpoint['student_model'])
    student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[rank])
    student_model.eval()

    val_dataset = SegCropDataset("/netscratch/zlu/dataset/youtubevos/sam2/pt_val/", augment=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, sampler=val_sampler, num_workers=4, pin_memory=True)

    evaluate(student_model, val_loader, teacher_backbone, device)

    cleanup_ddp()

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    #train_ddp(rank, world_size)
    eval_ddp(rank, world_size)


if __name__ == "__main__":
    
    main()
