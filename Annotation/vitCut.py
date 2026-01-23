import numpy as np
from sklearn.cluster import KMeans
import cv2
from torchvision.transforms.functional import resize, InterpolationMode
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from .crf import densecrf
from scipy import ndimage
from skimage import filters
from skimage import measure
from torchvision.ops import roi_align
from torchvision import transforms
from shapely.geometry import Polygon
from shapely.geometry import mapping

def bbox_from_mask(mask: np.ndarray):
    # bbox format is [x, y, width, height]
    x = np.where(mask.sum(axis=0))[0]
    y = np.where(mask.sum(axis=1))[0]
    bbox = [np.min(x), np.min(y), np.max(x) - np.min(x) + 1, np.max(y) - np.min(y) + 1]
    return np.array(bbox)


def num_corners_on_border_mask(mask):
    """
    :param mask: binary mask of shape (H, W)
    """
    # check if there is an overlap between the bbox and at list 2 image borders
    num_of_corners_on_border = mask[0, 0] + mask[0, -1] + mask[-1, 0] + mask[-1, -1]
    return num_of_corners_on_border

def kmeans_labeling(vector_groups, Ks=(2, 3)):
    """
    Performs K-means clustering on eigenvectors of each group in vector_groups. It returns a list of dictionaries
    where each dictionary contains the K-means labels, the group name, the eigenvector index and the K value.
    :param vector_groups: dictionary of the form {group_name: {eigenvectors: [eig_vec_1, eig_vec_2, ...]}}
    :param Ks: list of K values to use for K-means clustering
    """
    kmeans_labels = []
    
    for group_name, eig_vec_group in vector_groups.items():
        
        for i, eig_vec in enumerate(eig_vec_group["eigenvectors"]):
            # make sure eigen vector is numpy array
            v = np.array(eig_vec)
            dims = v.shape
            samples = v.reshape(dims[0] * dims[1], 1)
            for k in Ks:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(samples)
                labels = kmeans.labels_.reshape(dims)
                kmeans_labels.append({
                    'labels': labels,
                    'group_name': group_name,
                    'eig_vec': i+1,
                    'k': k
                })
    #print('#####',kmeans_labels)
    return kmeans_labels




def kmeans_labeling_list(kmeans_labels):
    """
    Takes the list of dictionaries of kmeans labels and returns a list of labels matrices.
    :param kmeans_labels:
    :return:
    """
    kmeans_labels_list = []
    for kmeans_label in kmeans_labels:
        kmeans_labels_list.append(kmeans_label['labels'])
    return kmeans_labels_list


def instances_from_semantic_labels(semantic_labels, min_mask_w=5, min_mask_h=5):
    """
    Performs instance segmentation from semantic labels of a single image. It takes the product of the "semantic" labels
    given by the K-means algorithm and outputs a list of instance masks as the connected components of each semantic
    label.
    :param semantic_labels: list of labels matrices of shape (H, W) for each patch in the image.
    :param min_mask_w: minimum width of a mask to be considered an instance
    :param min_mask_h: minimum height of a mask to be considered an instance
    """
    instance_masks = []
    for s_label in semantic_labels:
        dims = s_label.shape
        labels = np.unique(s_label)
        for l in labels:
            semantic_mask = (s_label == l).astype(np.uint8)
            if num_corners_on_border_mask(semantic_mask) >= 2:
                continue
            # break the mask into connected components
            components = cv2.connectedComponents(semantic_mask * 255, connectivity=4)[1]
            # put -1 on the background
            components[semantic_mask == 0] = -1
            # take on non-background connected components
            instance_labels = np.unique(components)[1:]
            if len(instance_labels) > 30:
                continue
            for i_label in instance_labels:
                instance_mask = np.zeros(dims)
                instance_mask[components == i_label] = 1
                bbox = bbox_from_mask(instance_mask)
                # if bbox is too small continue
                if bbox[2] < min_mask_w and bbox[3] < min_mask_h:
                    continue
                instance_masks.append(instance_mask)
    return instance_masks



def iou_between_masks(masks, device='cpu'):
    """
    Calculates the IoU between all pairs of masks in the input array of masks.
    :param masks: array of shape (N, H, W) where N is the number of masks
    :param device: device to use for the calculation
    :return: array of shape (N, N) with the IoU between all pairs of masks
    """
    masks_flat = masks.reshape(masks.shape[0], -1)
    # check if cuda available
    masks = torch.Tensor(masks_flat).to(device)
    with torch.no_grad():
        union_intersection_diff = torch.cdist(masks, masks, p=1.0)
        intersection = masks @ masks.T
        union = union_intersection_diff + intersection
        iou = intersection / union
        iou = iou.cpu().numpy()
    return iou


def IoU_bbox(mask1, mask2):
    """
    This method calculates the IoU between the two bboxes of mask1 and mask2.
    :param mask1:
    :param mask2:
    :return:
    """
    bbox_1 = bbox_from_mask(mask1)
    bbox_2 = bbox_from_mask(mask2)
    # calculate the intersection area
    x1 = max(bbox_1[0], bbox_2[0])
    y1 = max(bbox_1[1], bbox_2[1])
    x2 = min(bbox_1[0] + bbox_1[2], bbox_2[0] + bbox_2[2])
    y2 = min(bbox_1[1] + bbox_1[3], bbox_2[1] + bbox_2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    # calculate the union area
    union_area = bbox_1[2] * bbox_1[3] + bbox_2[2] * bbox_2[3] - intersection_area
    return intersection_area / union_area



def mask_post_processing(mask, image_rgb, device='cpu'):
    """
    Post-processing of the mask. It performs crf and returns the final mask in the original image size.
    In case of crf failure, it returns the original mask.
    mask: numpy array of shape [height, width] with [0,1] values
    image_rgb: PIL image
    return: tuple - (mask as numpy array of shape [height, width] with [0,1] values, success flag)
    """
    success = True
    image_orig_size = image_rgb.size
    rescale_size = (image_orig_size[1], image_orig_size[0])
    # resizes the mask to the original image size with nearest neighbor interpolation
    patches_mask = F.interpolate(torch.from_numpy(mask[None, None, :, :]), size=rescale_size, mode='nearest')[0][0].numpy()
    # crop the mask by the bounding box
    bbox = bbox_from_mask(patches_mask)
    crop_x = (max(bbox[0] - bbox[2]//3, 0), min((bbox[0] + bbox[2]) + bbox[2]//3, rescale_size[1]))
    crop_y = (max(bbox[1] - bbox[3]//3, 0), min((bbox[1] + bbox[3]) + bbox[3]//3, rescale_size[0]))
    mask_cropped = patches_mask[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    # crop the image by the bounding box
    img = np.asarray(image_rgb).copy()
    img_cropped = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1], :]
    # apply CRF to the bounding box
    try:
        pseudo_mask_crop = densecrf(img_cropped, mask_cropped)
        pseudo_mask_crop = ndimage.binary_fill_holes(pseudo_mask_crop >= 0.5)
        # create a pseudo mask with the same size as the original image
        pseudo_mask = np.zeros_like(patches_mask)
        pseudo_mask[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]] = pseudo_mask_crop
        # in case crf did not provide a mask or the IoU between the original mask and the pseudo mask is too different
        # we consider the mask as not an object
        if np.sum(pseudo_mask) == 0 or IoU_bbox(torch.from_numpy(patches_mask).to(device), torch.from_numpy(pseudo_mask).to(device)) < 0.5:
            return patches_mask, False
        binary_mask = pseudo_mask
    except Exception as e:
        # in case crf failed for some reason use the original mask
        binary_mask = patches_mask
        success = False
    return binary_mask, success

def mask_to_box(mask, image_rgb, device='cpu'):
    """
    Post-processing of the mask. It performs crf and returns the final mask in the original image size.
    In case of crf failure, it returns the original mask.
    mask: numpy array of shape [height, width] with [0,1] values
    image_rgb: PIL image
    return: tuple - (mask as numpy array of shape [height, width] with [0,1] values, success flag)
    """
    success = True
    image_orig_size = image_rgb.size
    rescale_size = (image_orig_size[1], image_orig_size[0])
    # resizes the mask to the original image size with nearest neighbor interpolation
    patches_mask = F.interpolate(torch.from_numpy(mask[None, None, :, :]), size=rescale_size, mode='nearest')[0][0].numpy()
    # crop the mask by the bounding box
    bbox = bbox_from_mask(patches_mask)
    x,y,w,h=bbox
    bbox=[x,y,x+w,y+h]
    return bbox

def resize_masks(masks, shape=(60,60)):
    # resize the eigenvectors to the largest patches shape using torchvision resize function
    for i, m in enumerate(masks):
        masks[i] = resize(torch.Tensor(m[None, :, :]), list(shape), interpolation=InterpolationMode.NEAREST)[0].numpy()
    return masks



def cluster_mask_by_iou(masks ,threshold=0.6, pivot_iter=5, device='cpu'):
    """
    This method performs clustering of masks based on the IoU between them. It is greedy algorithm that tries to find
    the largest clusters of masks by IoU distance between them. The return value is a list of clusters dictionaries
    that contain the indices of the masks that belong to the cluster and other information about the cluster.
    :param masks: numpy array of shape (N, H, W) where N is the number of masks
    :param threshold: IoU threshold. Above this threshold masks are considered to be in the same cluster.
    :param pivot_iter: number of iterations to find the pivot mask. The pivot mask is the mask that has the most
    number of masks close masks above the threshold.
    """
    ious = iou_between_masks(masks, device=device)
   
    max_iou_indices = np.where(ious > threshold)[0]
    # get the common indices that are above the threshold
    indices, occurrences = np.unique(max_iou_indices, return_counts=True)
    # sort the indices by the number of occurrences in descending order
    occurrences_sorted = np.argsort(occurrences)[::-1]
    indices = indices[occurrences_sorted]

    ind_list = indices.tolist()
    indices_to_keep = []
    selected_mask = []
    used_indices = set()
    for j in range(len(indices)):
        if len(ind_list) == 0:
            break
        pivot_index = ind_list.pop(0)
        segment_cluster_indices = np.array([pivot_index]).astype(int)
        for iter in range(pivot_iter):
            prev_ind_num = len(segment_cluster_indices)
            above_threshold = np.array(np.where(ious[pivot_index] > threshold)).flatten()
            segment_cluster_indices = np.unique(np.concatenate((segment_cluster_indices, above_threshold)))
            if prev_ind_num == len(segment_cluster_indices):
                break
            ious_between_proposals = ious[:, segment_cluster_indices]
            ious_between_proposals = ious_between_proposals[segment_cluster_indices, :]
            # sum over the columns
            ious_sum = np.sum(ious_between_proposals, axis=1)
            pivot_index = segment_cluster_indices[np.argmax(ious_sum)]
        if pivot_index not in used_indices:
            # get rid of all indices in segment_cluster_indices that are also in used_indices
            segment_cluster_indices = np.array([i for i in segment_cluster_indices if i not in used_indices])
            pivot_mask = masks[pivot_index]
            pivot_bbox = bbox_from_mask(pivot_mask)
            selected_mask.append({
                'pivot_index': pivot_index,
                'pivot_mask': pivot_mask,
                'pivot_bbox': pivot_bbox,
                'segment_cluster_indices': segment_cluster_indices,
                'cluster_size': len(segment_cluster_indices),
            })
            indices_to_keep.append(pivot_index)
        used_indices.update(segment_cluster_indices)
        # remove the indices of the current mask from the list
        ind_list = [i for i in ind_list if i not in used_indices]

    return selected_mask


def clustering_getbox(masks_proposals,
                   image_rgb:Image.Image=None,
                   tau_m=0.2,
                   patches_shape=(60, 60),
                   max_masks_per_img=10,
                   device='cpu'):
    """
    This method performs the IoU clustering including CRF post-processing on the masks.
    It returns a list of dictionaries that represent the final votecut objects in the image.
    :param masks_proposals: list of numpy arrays of shape (H, W) where H and W are the height and width of the masks
    :param image_rgb: PIL image
    :param tau_m: The threshold for the final mask "Pixel-wise" voting
    :param patches_shape: The shape of the patches to use for resizing the masks before iou clustering
    :param max_masks_per_img: The maximum number of masks to return per image
    :param device:
    :return: list of dictionaries of the form: {"bit_mask": the mask after "Pixel-wise" voting, "mask": the final mask
    after CRF post-processing, "crf_success": indicates whether CRF post-processing succeeded, "cluster_size": The size
    of the cluster of masks that the mask belongs to.}
    """
    # make sure the masks are of the same size
    
    masks_proposals = resize_masks(masks_proposals, shape=patches_shape)
    
    # create numpy array from the masks list of numpy arrays
    masks = np.array(masks_proposals)
    
    boxes = []
    
    mask_clusters = cluster_mask_by_iou(masks, threshold=0.6, device=device)
    image_boxes = []

    # Step 4: for each cluster, do pixel voting, CRF, then get bbox
    for i, cluster_data in enumerate(mask_clusters):
        if len(image_boxes) >= max_masks_per_img or i >= 100:
            break

        cluster_size = cluster_data['cluster_size']
        cluster_masks = masks[cluster_data['segment_cluster_indices']]

        # Voting
        mask = np.sum(cluster_masks, axis=0) / cluster_masks.shape[0]
        bit_mask = (mask > tau_m).astype(np.uint8)

        # CRF post-processing
        #final_mask, success = mask_post_processing(bit_mask, image_rgb, device=device)
        bbox=mask_to_box(bit_mask, image_rgb, device=device)
        #print('bbox_test',bbox_test)
        '''
        if final_mask is None or final_mask.max() == 0:
            continue

        polygon, bbox = binary_mask_to_polygon(final_mask)
        if polygon==None or bbox == None:
                continue
        print('changed:',bbox)
        '''
        image_boxes.append({
            "bbox": bbox,
            "cluster_size": cluster_size
        })

    # Sort by cluster size
    out = sorted(image_boxes, key=lambda x: x['cluster_size'], reverse=True)
    
    out = [
        box['bbox'] for box in out
        if not is_bbox_near_image_size(box['bbox'], image_rgb)
        ]
    
    return out

def masks_to_polygons(mask,padding=None):

    contours_thresholhold = 500
    cannot_visualize_no_mask = False
    masks = [mask[i, :,:] for i in range(mask.shape[0])]
    
    mask = masks[0]
   
    mask = (mask > 0).astype(np.uint8)
            #finding contours through Skimage function
    
    contours = measure.find_contours(mask, 0.5)
        #confidence_score = confidence_score.cpu().numpy().squeeze()
            #Taking the biggest of the contour since we know single mask is required for each bbox
    contour = max(contours, key=len)

            #Flipping so x axis have first value
    contour = np.float32(np.flip(contour, axis=1))
    
            # Approximate contour if it has more than contours_thresholhold points
    if len(contour) > contours_thresholhold:
                
        epsilon = 0.0005 * cv2.arcLength(contour, True)
                # print(f"EPSILON VALUE : {epsilon} AND CURRENT LENGTH OF CONTOUR : {len(contour)}\n\n")
        contour = cv2.approxPolyDP(contour, epsilon, True)
    
    x, y, w, h = cv2.boundingRect(contour)
     
    contour = np.squeeze(contour)
    #print(contour)
    if padding!=None:
        left,top = padding[0],padding[1]
        bbox = (x-left, y-top, x + w, y + h)  
        contour = np.array(contour) - np.array([left, top])
    else:
        bbox = (x, y, x + w, y + h)  
    polygon_np = np.array(contour, dtype=np.int32).tolist()
    points_str = ";".join(
            f"{point[0]},{point[1]}" if len(point) == 2 else f"{point[0][0]},{point[0][1]}" for point in polygon_np)

  
        
    return points_str,bbox

def get_Boxes(image_rgb, eig_vec_groups, Ks=(2, 3), tau_m=0.2,device='cuda'):
    
    kmeans_labels = kmeans_labeling(eig_vec_groups, Ks=Ks)
    masks_proposals = instances_from_semantic_labels(kmeans_labeling_list(kmeans_labels))
    boxes = clustering_getbox(masks_proposals, image_rgb, tau_m=tau_m, device=device)
    return boxes 
    
def get_bbox_corners(box):
    x_min, y_min, x_max, y_max = box
    return np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ], dtype=np.float32)
    
def warp_bbox_by_flow(box, flow):
    corners = get_bbox_corners(box)

    h, w = flow.shape[:2]

    warped_corners = []
    for (x, y) in corners:
        # 保证索引在图像范围内，防止越界
        px = min(max(int(round(x)), 0), w-1)
        py = min(max(int(round(y)), 0), h-1)

        flow_at_point = flow[py, px]  # 注意 y 是行，x 是列
        warped_point = np.array([x, y]) + flow_at_point
        warped_corners.append(warped_point)

    warped_corners = np.array(warped_corners)
    x_min_warp = warped_corners[:, 0].min()
    y_min_warp = warped_corners[:, 1].min()
    x_max_warp = warped_corners[:, 0].max()
    y_max_warp = warped_corners[:, 1].max()

    return [int(round(x_min_warp)),
        int(round(y_min_warp)),
        int(round(x_max_warp)),
        int(round(y_max_warp))]

def roi_align_on_image(image_rgb, bboxes_list, output_size=224):
    """
    image_tensor: [B, C, H, W]
    bboxes: List of [x1, y1, x2, y2] for each ROI, as Tensor of shape [N, 4]
    output_size: int or tuple, like 224 or (224, 224)
    """
    transform = transforms.ToTensor()
    image_tensor = transform(image_rgb).unsqueeze(0)  #[1, 3, H, W]]
    bboxes = torch.tensor(bboxes_list, dtype=torch.float32)  # shape: [N, 4]
    roi_indices = torch.zeros((bboxes.shape[0], 1), dtype=torch.float32)
    rois = torch.cat([roi_indices, bboxes], dim=1)  # [N, 5]

    # Step 3: ROIAlign
    crops = roi_align( 
        input=image_tensor,
        boxes=rois,
        output_size=224,
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=True
    )
    return crops  # [N, C, output_size, output_size]

def mask2polygon(final_masks, blur_ksize=5, mask_thresh=0.1):
    polygons = []
    bboxes = []
    for mask in final_masks:
        
        mask_uint8 = (mask > mask_thresh).astype(np.uint8) * 255
        blurred = cv2.GaussianBlur(mask_uint8, (blur_ksize, blur_ksize), 0)
        _, smooth_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

        
        contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        #print(contours)
        if max_contour.shape[0] > 4:
            # 保存 polygon
            #print(max_contour.reshape(-1, 2))
            polygons.append(max_contour.reshape(-1, 2))
            # 计算 bbox
            #x, y, w, h = cv2.boundingRect(max_contour)
            #bboxes.append([x, y, x + w, y + h])
    
    return polygons

def map_masks_to_original_image(image_rgb,pred_masks,fused_box,mask_threshold=0.1):
    N = pred_masks.shape[0]
    orig_w, orig_h = image_rgb.size 
    full_masks = np.zeros((N, orig_h, orig_w), dtype=np.float32)

    # 上采样掩码到bbox大小
    for i in range(N):
        mask = pred_masks[i]  # [ 64, 64]
        if isinstance(mask, np.ndarray):
            raise TypeError(f"Expected torch.Tensor but got numpy.ndarray at index {i}")

        if mask.ndim == 2:  # [h,w]
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        if mask.dim() == 3:
            mask = mask.unsqueeze(0) 
        
        mask = (mask > 0.3).float()   
        x1, y1, x2, y2 = map(int, fused_box[i])
        box_w = x2 - x1
        box_h = y2 - y1
        

        # 使用双线性插值把56x56掩码resize到bbox大小
        mask_resized = F.interpolate(mask, size=(box_h, box_w), mode='nearest') #torch.Size([1, 1, h, w])
        #print('mask_resized',mask_resized.size())
        mask_resized = mask_resized.squeeze()
        mask_resized = mask_resized.cpu().numpy()
        
        # 限制坐标范围，防止越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(orig_w, x2)
        y2 = min(orig_h, y2)
        
        # 放置 到全图掩码
        full_masks[i, y1:y2, x1:x2] = mask_resized[:y2 - y1, :x2 - x1]
        
        '''
        mask_min = full_masks.min().item()
        mask_max = full_masks.max().item()
        mask_mean = full_masks.mean().item()
        print(f"full_masks: min={mask_min:.4f}, max={mask_max:.4f}, mean={mask_mean:.4f}")
        '''
   
    return full_masks


def binary_mask_to_polygon(binary_mask):
    tolerance = 1
    binary_mask = (binary_mask > 0).astype(np.uint8)
    #print(binary_mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    if not contours :
        #print('inzo....')
        return None,None
    polygons = []
    contour = max(contours, key=len)
    
    #points = [f"{point[0][0]},{point[0][1]}" for point in contour]
    contour_tuple = [tuple(point[0]) for point in contour]
    #print('length::::',len(contour_tuple))
    if len(contour_tuple) < 20:
        return None,None
    area = cv2.contourArea(contour)
    
    polygon = Polygon(contour_tuple)
    simplified_polygon = polygon.simplify(tolerance, preserve_topology=True)
    simplified_coords = list(mapping(simplified_polygon)["coordinates"])
    simplified_data = list(simplified_coords[0])
    if len(simplified_data) > 20:
        #print(simplified_data)
        polygon_string = ";".join([f"{int(x)},{int(y)}" for x, y in simplified_data])
        #polygon_string = ";".join(simplified_data) + ";"  
        #polygons.append(polygon_string)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if area < 5000 or aspect_ratio > 5 or aspect_ratio < 0.2:
            return None,None
        bbox = (x, y, x + w, y + h) 
        return polygon_string,bbox
    else:
        return None,None
    #print('++',len(simplified_data))

def smooth_mask(mask, ksize=5):
    """
    mask: 0/1 numpy 数组 (二值分割结果)
    ksize: 模糊核大小，越大越平滑
    """
    mask = (mask.astype(np.uint8) * 255)
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    _, smooth = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    return smooth // 255  # 返回 0/1 mask


def is_bbox_near_image_size(bbox, image, threshold=0.95):
    img_w, img_h = image.size
    x_min, y_min, x_max, y_max = bbox

    box_w = x_max - x_min
    box_h = y_max - y_min
    box_area = box_w * box_h
    img_area = img_w * img_h

    coverage = box_area / img_area
    return coverage >= threshold