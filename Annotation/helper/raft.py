import sys
sys.path.append("/netscratch/zlu/RAFT") 
sys.path.append("/netscratch/zlu/RAFT/core")
#from raft import RAFT
from core.raft import RAFT
from core.utils.utils import InputPadder 
from PIL import Image,ImageDraw
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from votecut.videocut import mask_post_processing
import random
def init_raft_model(model_path,args,device='cuda'):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model = model.module.to(device)
    model.eval()
    return model


def compute_flow(raft_model, image1, image2):
    # image1, image2: [1, 3, H, W]
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    flow_low, flow_up = raft_model(image1, image2, iters=20, test_mode=True)
    flow_up = padder.unpad(flow_up) 
    return flow_up 

def get_flow(model, img1_PILRGB, img2_PILRGB,device='cuda'):
    #img1_changed = img1_PILRGB.resize((1920, 1080))
    #img2_changed = img2_PILRGB.resize((1920, 1080))

    img1 = np.array(img1_PILRGB).astype(np.float32) / 255.0
    img2 = np.array(img2_PILRGB).astype(np.float32) / 255.0
    #  NumPy to tensor
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)
    
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    
    with torch.no_grad():
        flow_low, flow_up = model(img1, img2, iters=6, test_mode=True)
    
    flow_up = padder.unpad(flow_up)
    #flow_up = torch.nn.functional.interpolate(flow_up, size=img1_PILRGB.size[::-1], mode='bilinear', align_corners=False) * (img1_PILRGB.size[0]/1024)

    return flow_up

def check_frame_static(flow,threshold=2,ratio=0.95):
    #threshold 5 pixel moving
    #ratio 95%of all frame pixel
    flow = flow[0].permute(1, 2, 0).cpu().numpy() 
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.sqrt(u**2 + v**2)
    static_mask = magnitude < threshold
    static_ratio = np.sum(static_mask) / static_mask.size
    #print(static_ratio,"mean_magnitude:", np.mean(magnitude),"max_magnitude:", np.max(magnitude))
    return (static_ratio > ratio),np.mean(magnitude)

def wrap_mask_inlist(img_info,flow,current_img,device):
    valid_img_info = []
    frame_static,flow_mean = check_frame_static(flow)
    #print('frame_static:',frame_static)
    #random_number = random.randint(1, 10)
    for data in img_info:
        if frame_static:
            flow_threshold=flow_mean
        else:
            flow_threshold=1.5
        wraped_mask,bbox,is_background = warp_mask_with_flow_origin(data['mask'], flow,current_img,flow_threshold,device)
        #print('is_background##:',is_background)
        
        if flow.dim() == 4 and flow.shape[0] == 1 and flow.shape[1] == 2:
            flow_check = flow.squeeze(0).permute(1, 2, 0)
        if not filter_by_flow_magnitude(wraped_mask, flow_check, max_thresh=80):
            continue
       
        
        
        if wraped_mask is not None:
            #data["before"]=data["box"]
            data["mask"]=wraped_mask
            data["box"] = bbox
            data["background"]=is_background
            data["weight"]=0.5
            data['is_current'] = False
            valid_img_info.append(data)
            #print(data["box"])
            #print('before:',data["before"],bbox)
    
    #overlay_mask_on_image(valid_img_info,current_img, "/netscratch/zlu/test/cutvler/wrapped_output"+str(random_number)+".jpg")
    return valid_img_info

def filter_by_flow_magnitude(mask, flow, max_thresh=30.0):
    
    flow_mag = torch.linalg.norm(flow, dim=2)  # 计算每个像素光流大小
    masked_mag = flow_mag[mask > 0.5]
    
    #print("Sample flow vectors:", masked_mag[:100]) 
    if masked_mag.numel() == 0:
        return False

    mean_mag = masked_mag.mean()
    #print("Mean flow magnitude:", mean_mag)
   
    return (mean_mag < max_thresh)




def filter_by_flow_gradient(mask, flow, gradient_thresh=50):
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    grad_x_x = torch.abs(flow_x[:, 1:] - flow_x[:, :-1])
    grad_y_x = torch.abs(flow_x[1:, :] - flow_x[:-1, :])
    grad_x_y = torch.abs(flow_y[:, 1:] - flow_y[:, :-1])
    grad_y_y = torch.abs(flow_y[1:, :] - flow_y[:-1, :])

    # 补0让梯度大小与flow一致
    pad_x = torch.zeros((flow.shape[0], 1), device=flow.device)
    pad_y = torch.zeros((1, flow.shape[1]), device=flow.device)

    grad_x_x = torch.cat([grad_x_x, pad_x], dim=1)
    grad_y_x = torch.cat([grad_y_x, pad_y], dim=0)
    grad_x_y = torch.cat([grad_x_y, pad_x], dim=1)
    grad_y_y = torch.cat([grad_y_y, pad_y], dim=0)

    flow_grad_mag = grad_x_x + grad_y_x + grad_x_y + grad_y_y

    masked_grad = flow_grad_mag[mask > 0.5]

    if masked_grad.numel() == 0:
        return False

    mean_grad = masked_grad.mean()
    #print("Max flow gradient:", mean_grad)
    return mean_grad < gradient_thresh




def warp_mask_with_flow(ref_mask, flow,current_img,device):
    upsample_scale = 2
    eps = 1e-6  
    flow_threshold = 1.0
    assert isinstance(flow, torch.Tensor)
    assert flow.shape[0] == 1 and flow.shape[1] == 2, "flow must be of shape [1, 2, H, W]"
    flow = flow.to(device)
    _, _, H, W = flow.shape

    is_background=False
    ref_mask_tensor = torch.from_numpy(ref_mask).unsqueeze(0).unsqueeze(0).float().to(device)
    ref_mask_tensor_upsampled = F.interpolate(ref_mask_tensor, scale_factor=upsample_scale, mode='bilinear', align_corners=False)
    up_H, up_W = ref_mask_tensor_upsampled.shape[-2:]

 
    flow_upsampled = F.interpolate(flow, size=(up_H, up_W), mode='bilinear', align_corners=False) * upsample_scale
    u = flow_upsampled[0, 0]
    v = flow_upsampled[0, 1]
    #check is background
    
    flow_vec = torch.stack((u, v), dim=-1)  # shape: [up_H, up_W, 2]
    mask_bool = ref_mask_tensor_upsampled[0, 0] > 0.5  # [up_H, up_W]
    
    if mask_bool.sum() > 0:
        masked_flow = flow_vec[mask_bool]  # [N, 2]
        flow_magnitude = masked_flow.norm(dim=1)
        mean_flow = flow_magnitude.mean()
        
        if mean_flow.item() < flow_threshold:
            is_background = True
    
      
    grid_y, grid_x = torch.meshgrid(
        torch.arange(up_H, device=device), 
        torch.arange(up_W, device=device), 
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).float()

    target_coords = grid - torch.stack((u, v), dim=0)

    norm_x = (target_coords[0] + 0.5) / up_W * 2 - 1
    norm_y = (target_coords[1] + 0.5) / up_H * 2 - 1

    norm_x = norm_x.clamp(-1 + eps, 1 - eps)
    norm_y = norm_y.clamp(-1 + eps, 1 - eps)

    grid_norm = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0).to(dtype=ref_mask_tensor_upsampled.dtype)

    # 4. warp
    tgt_mask_tensor = F.grid_sample(
        ref_mask_tensor_upsampled, 
        grid_norm, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=False
    )

 
    tgt_mask_tensor_down = F.interpolate(tgt_mask_tensor, size=(H, W), mode='nearest')

    tgt_mask = tgt_mask_tensor_down.squeeze().cpu().numpy()
    tgt_mask = cv2.GaussianBlur(tgt_mask, (5, 5), sigmaX=2)
    tgt_mask_binary = (tgt_mask > 0.5).astype(np.uint8)

    if tgt_mask_binary.sum() == 0:
        return None, None,False

    contours, _ = cv2.findContours(tgt_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, x + w, y + h)
    else:
        return None, None,False

    return tgt_mask_binary, bbox,is_background

def warp_ref_frame_tensor(ref_img_tensor, flow):
    """
    ref_img_tensor: [1, 3, H, W]
    flow: [1, 2, H, W]  # output from RAFT
    """
    B, C, H, W = ref_img_tensor.shape

    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(ref_img_tensor.device)  # [H, W, 2]

    # Add flow
    flow = flow[0].permute(1, 2, 0)  # [H, W, 2]
    new_locs = grid + flow  # [H, W, 2]

    # Normalize to [-1,1]
    new_locs[..., 0] = 2.0 * new_locs[..., 0] / (W - 1) - 1.0
    new_locs[..., 1] = 2.0 * new_locs[..., 1] / (H - 1) - 1.0

    new_locs = new_locs.unsqueeze(0)  # [1, H, W, 2]

    # grid_sample expects grid in [B, H, W, 2]
    warped = F.grid_sample(ref_img_tensor, new_locs, align_corners=True)
    return warped  # [1, 3, H, W]


def overlay_mask_on_image(
    valid_img_info, 
    current_img: Image.Image, 
    output_path: str, 
    box_color=(255, 0, 0),         # current box 的颜色（红）
    before_box_color=(0, 0, 255),  # before box 的颜色（蓝）
    box_width=2
):
    current_img = current_img.convert("RGB")
    draw = ImageDraw.Draw(current_img)

    for data in valid_img_info:
        # 绘制原始框（before）
        if "before" in data and data["before"] is not None:
            x1, y1, x2, y2 = map(int, data["before"])
            draw.rectangle([x1, y1, x2, y2], outline=before_box_color, width=box_width)
        
        # 绘制更新后的框（box）
        if "box" in data and data["box"] is not None:
            x1, y1, x2, y2 = map(int, data["box"])
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
        
    current_img.save(output_path, format="JPEG")
    print(f"wrap 后的框可视化已保存至：{output_path}")


def warp_features_with_flow(features, flow):
    """
    features: [B, D, H, W]
    flow: [B, 2, H, W]
    """
    B, C, H_feat, W_feat = features.shape
    _, _, H_flow, W_flow = flow.shape

    flow_down = F.interpolate(flow, size=(H_feat, W_feat), mode='bilinear', align_corners=False)  # [1, 2, 30, 30]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H_feat, device=features.device),
        torch.linspace(-1, 1, W_feat, device=features.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [H_feat, W_feat, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H_feat, W_feat, 2]

    flow_norm = torch.zeros_like(flow_down)
    flow_norm[:, 0, :, :] = flow_down[:, 0, :, :] / ((W_feat - 1) / 2)
    flow_norm[:, 1, :, :] = flow_down[:, 1, :, :] / ((H_feat - 1) / 2)

    
    grid = grid + flow_norm.permute(0, 2, 3, 1)  # [B, H_feat, W_feat, 2]

    warped = F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=False)
    #print('warped',warped.size())
    return warped  # [1, 384, 30, 30]


def flow_vis(flow, image, output_path, overlay=True):
    h, w = flow.shape[:2]
    flow = flow.copy()
    
    # 计算光流大小和角度
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)

    # 构造HSV图像
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)   # Hue (方向)
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = np.clip((magnitude / magnitude.max()) * 255, 0, 255).astype(np.uint8)  # Value (强度)

    # 转换为BGR图像（OpenCV使用BGR）
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 保存光流图
    cv2.imwrite(output_path, flow_bgr)


def flow_to_image(img, flow, step=16, scale=1.0, color=(0, 255, 0)):
    """
    在图像上绘制光流箭头
    :param img: 背景图像，HWC格式 (uint8)
    :param flow: 光流数组，(H, W, 2)，单位是像素位移
    :param step: 每隔多少像素画一个箭头（控制密度）
    :param scale: 箭头长度缩放比例
    :param color: 箭头颜色（B, G, R）
    :return: 带箭头的图像
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    vis = img.copy()

    # 生成网格坐标
    y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(int)
    fx, fy = flow[y, x, 0], flow[y, x, 1]

    for i in range(len(y)):
        for j in range(len(x[0])):
            pt1 = (x[i, j], y[i, j])
            dx, dy = fx[i, j], fy[i, j]
            pt2 = (int(pt1[0] + scale * dx), int(pt1[1] + scale * dy))
            cv2.arrowedLine(vis, pt1, pt2, color=color, thickness=1, tipLength=0.3)

    return vis


def warp_mask_with_flow_origin(ref_mask, flow, current_img, flow_threshold,device):
    eps = 1e-6  
    

    assert isinstance(flow, torch.Tensor)
    assert flow.shape[0] == 1 and flow.shape[1] == 2, "flow must be of shape [1, 2, H, W]"
    flow = flow.to(device)
    _, _, H, W = flow.shape

    is_background = False

    # Prepare mask tensor (no upsampling!)
    ref_mask_tensor = torch.from_numpy(ref_mask).unsqueeze(0).unsqueeze(0).float().to(device)  # shape: [1, 1, H, W]

    # Flow components (no upsampling!)
    u = flow[0, 0]  # [H, W]
    v = flow[0, 1]  # [H, W]
    flow_vec = torch.stack((u, v), dim=-1)  # [H, W, 2]

    # Create mask_bool directly from original mask
    mask_bool = ref_mask_tensor[0, 0] > 0.5  # [H, W]

    if mask_bool.sum() > 0:
        masked_flow = flow_vec[mask_bool]  # [N, 2]
        flow_magnitude = masked_flow.norm(dim=1)
        #dynamic_thresh = torch.quantile(flow_magnitude, 0.25).item()
        #print('dynamic_thresh',dynamic_thresh)
        mean_flow = flow_magnitude.mean()
        #print('mean_flow',mean_flow)
        if mean_flow.item() < flow_threshold:
            is_background = True

    # Build sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, H, W]

    target_coords = grid - torch.stack((u, v), dim=0)  # subtract flow: [2, H, W]

    norm_x = (target_coords[0] + 0.5) / W * 2 - 1
    norm_y = (target_coords[1] + 0.5) / H * 2 - 1
    norm_x = norm_x.clamp(-1 + eps, 1 - eps)
    norm_y = norm_y.clamp(-1 + eps, 1 - eps)

    grid_norm = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Warp mask
    tgt_mask_tensor = F.grid_sample(
        ref_mask_tensor, 
        grid_norm, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=False
    )

    # Get result mask
    tgt_mask = tgt_mask_tensor.squeeze().cpu().numpy()  # shape: [H, W]
    tgt_mask = cv2.GaussianBlur(tgt_mask, (5, 5), sigmaX=2)
    tgt_mask_binary = (tgt_mask > 0.5).astype(np.uint8)

    if tgt_mask_binary.sum() == 0:
        return None, None, False

    contours, _ = cv2.findContours(tgt_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, x + w, y + h)
    else:
        return None, None, False

    return tgt_mask_binary, bbox, is_background



