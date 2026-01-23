import numpy as np
from sklearn.cluster import KMeans
import cv2
from torchvision.transforms.functional import resize, InterpolationMode
import torch
import torch.nn.functional as F
from PIL import Image
from .crf import densecrf
from scipy import ndimage
from skimage import filters
from typing import List
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

def kmeans_video_labeling(vector_groups, Ks=(2, 3)):
    """
    Performs K-means clustering on eigenvectors of each group in vector_groups. It returns a list of dictionaries
    where each dictionary contains the K-means labels, the group name, the eigenvector index and the K value.
    :param vector_groups: dictionary of the form {group_name: {eigenvectors: [eig_vec_1, eig_vec_2, ...]}}
    :param Ks: list of K values to use for K-means clustering
    """
    kmeans_labels = []
    for img_eig in vector_groups:
        if img_eig is vector_groups[0]:
            frame = 0  #current frame
        else:
            frame = 1 #ref frame
        for group_name, eig_vec_group in img_eig.items():
            
            for i, eig_vec in enumerate(eig_vec_group["eigenvectors"]):
                # make sure eigen vector is numpy array
                v = np.array(eig_vec)
                dims = v.shape
                samples = v.reshape(dims[0] * dims[1], 1)
                for k in Ks:
                    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(samples)
                    labels = kmeans.labels_.reshape(dims)
                    kmeans_labels.append({
                        'frame': frame,
                        'labels': labels,
                        'group_name': group_name,
                        'eig_vec': i+1,
                        'k': k
                    })
    
    return kmeans_labels


def kmeans_vediolabeling_list(kmeans_labels):
    """
    Takes the list of dictionaries of kmeans labels and returns a list of labels matrices.
    :param kmeans_labels:
    :return:
    """
    kmeans_labels_list = []
    frame_list =[]
    for kmeans_label in kmeans_labels:
        frame_list.append(kmeans_label['frame'])
        kmeans_labels_list.append(kmeans_label['labels'])
    return kmeans_labels_list,frame_list


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



def videoinstances_from_semantic_labels(semantic_labels, frame_list,min_mask_w=5, min_mask_h=5):
    """
    Performs instance segmentation from semantic labels of a single image. It takes the product of the "semantic" labels
    given by the K-means algorithm and outputs a list of instance masks as the connected components of each semantic
    label.
    :param semantic_labels: list of labels matrices of shape (H, W) for each patch in the image.
    :param min_mask_w: minimum width of a mask to be considered an instance
    :param min_mask_h: minimum height of a mask to be considered an instance
    """
    instance_masks = []
    for s_label,frame in zip(semantic_labels,frame_list):
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
                instance_masks.append({'frame': frame,
                            'instance_mask':instance_mask
                            })
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


def cluster_mask_by_iou(masks ,threshold=0.5, pivot_iter=5, device='cpu'):
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
        #print(selected_mask)
    return selected_mask

def video_cluster_mask_by_iou(masks ,threshold=0.5, pivot_iter=5, device='cpu'):
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


def resize_masks(masks, shape=(60,60)):
    # resize the eigenvectors to the largest patches shape using torchvision resize function
    for i, m in enumerate(masks):
        masks[i] = resize(torch.Tensor(m[None, :, :]), list(shape), interpolation=InterpolationMode.NEAREST)[0].numpy()
    return masks


def iou_clustering(img_info,
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
    
    masks = [item["mask"] for item in img_info if "mask" in item]
    masks = np.array(masks)
    # cluster the masks by IoU
    mask_clusters = video_cluster_mask_by_iou(masks, threshold=0.6, device=device)
    # perform post-processing on the masks
    image_masks = []
    for i, cluster_data in enumerate(mask_clusters):
        if len(image_masks) >= max_masks_per_img or i >= 100:
            break
        cluster_size = cluster_data['cluster_size']
        cluster_masks = masks[cluster_data['segment_cluster_indices']]
        # we consider a patch as belonging to an object if at least tau_m percent of the clusters masks agree on it
        mask = np.sum(cluster_masks, axis=0)/cluster_masks.shape[0]
        bit_mask = (mask > tau_m).astype(np.uint8)
        final_mask, success = mask_post_processing(bit_mask, image_rgb, device=device)
        image_masks.append({
            "bit_mask": bit_mask,
            "mask": final_mask,
            "crf_success": success,
            "cluster_size": cluster_size,
        })
        #print(image_masks)
    out = sorted(image_masks, key=lambda x: x['cluster_size'], reverse=True)
    
    return out


def clustering(img_info,
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
    
    masks = [item["mask"] for item in img_info if "mask" in item]
    # create numpy array from the masks list of numpy arrays
    masks = np.array(masks_proposals)
    
    if not filtered_mask:  # Handle empty results
        return []
    image_masks = []
    for mask in filtered_mask:
        bit_mask = (mask > tau_m).astype(np.uint8)
        
        final_mask, success = mask_post_processing(bit_mask, image_rgb, device=device)
        
        image_masks.append( {
            "bit_mask": bit_mask,
            "mask": final_mask,
            "crf_success": success,
            
        })

    
    return image_masks

def clustering_getbox_mask(masks_proposals,
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
    w,h = image_rgb.size
    
    image_masks = []
    mask_clusters = cluster_mask_by_iou(masks, threshold=0.6, device=device)
    rescale_size = (h, w)  # H, W
    for i, cluster_data in enumerate(mask_clusters):
        
        cluster_size = cluster_data['cluster_size']
        cluster_masks = masks[cluster_data['segment_cluster_indices']]
        # we consider a patch as belonging to an object if at least tau_m percent of the clusters masks agree on it
        mask = np.sum(cluster_masks, axis=0)/cluster_masks.shape[0]
        
        bit_mask = (mask > tau_m).astype(np.uint8)
        patches_mask = F.interpolate(torch.from_numpy(bit_mask[None, None, :, :]), size=rescale_size, mode='nearest')[0][0].numpy()
        bbox = bbox_from_mask(patches_mask)
        bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        if is_bbox_over_threshold(bbox,image_rgb) ==False :
            image_masks.append({
                "bit_mask": patches_mask,
                "mask":patches_mask,
                "box":bbox
            })
   
    return image_masks

def get_Masks(image_rgb, eig_vec_groups, Ks=(2, 3), tau_m=0.2,device='cuda'):
    kmeans_labels = kmeans_labeling(eig_vec_groups, Ks=Ks)
    masks_proposals = instances_from_semantic_labels(kmeans_labeling_list(kmeans_labels))
    image_masks = clustering_getbox_mask(masks_proposals, image_rgb, tau_m=tau_m, device=device)
    return image_masks



def Video_cut(image_rgb, eig_vec_groups, Ks=(2, 3), tau_m=0.2, device='cpu'):
    """
    This method performs the votecut algorithm on the image. It takes the eigenvectors of the image and performs votecut
    pipeline on them. image_rgb is used to perform CRF and recover the correct scale for the image.
    It returns a list of dictionaries that represent the votecut objects in the image.
    :param image_rgb: PIL image
    :param eig_vec_groups: dictionary of the form {group_name: {eigenvectors: [eig_vec_1, eig_vec_2, ...]}}
    :param Ks: list of K values to use for K-means clustering
    :param tau_m: The threshold for the final mask "Pixel-wise" voting
    :param device:
    :return:
    """

    kmeans_labels = kmeans_labeling(eig_vec_groups, Ks=Ks)
    # get contiguous masks from kmeans labels
    masks_proposals = instances_from_semantic_labels(kmeans_labeling_list(kmeans_labels))
    #print('#######',masks_proposals)
    # perform iou_clustering on the masks
    image_masks = iou_clustering(masks_proposals, image_rgb, tau_m=tau_m, device=device)
    #print('+++',image_masks)
    return image_masks





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


def group_by_iou(data_list, iou_threshold=0.6,max_masks_per_img=9):
   
    #boxes = [item["box"] for item in data_list if "box" in item]
    #group the simular bounding box
    groups = []
    visited = set()
    for i in range(len(data_list)):
        if "box" not in data_list[i]:
            continue 
        if i in visited:
            continue
        current_group = [data_list[i]]
        visited.add(i)
        for j in range(i+1, len(data_list)):
            if "box" not in data_list[j]:
                continue 
            if j in visited:
                continue
            iou = compute_iou(data_list[i]["box"], data_list[j]["box"])
            if iou > iou_threshold:
                current_group.append(data_list[j])
                visited.add(j)
        #if len(current_group) > 1:
        groups.append(current_group)
    groups = sorted(groups, key=lambda g: len(g), reverse=True) 
    
    if len(groups)>max_masks_per_img: 
        #groups = [g for g in groups if len(g) > 1]
        groups = groups[:max_masks_per_img]

    for i, group in enumerate(groups):
        print(f"Group {i+1}: {len(group)} boxes")
    return groups

#def process_mask_group(mask_group,img, tau_m=0.5,device='cpu'):
    


def iou_mask_grouping_cluster(data_list, img,iou_thresh=0.6, tau_m=0.5,device='cpu'):
    groups = group_by_iou(data_list, iou_thresh)
    results = []
    for group in groups:
        
        cluster_masks = np.array([item['bit_mask'] for item in group])  # shape: (N, H, W)
        mask = np.sum(cluster_masks, axis=0)/cluster_masks.shape[0]
        #print(np.unique(mask))
        mask = (mask > tau_m).astype(np.uint8) 

        #bit_mask = (mask > tau_m).astype(np.uint8)
        #bbox = bbox_from_mask(mask)
        #final_mask, success = mask_crf_processing(mask,bbox, img, device=device)
        final_mask, success = mask_post_processing(mask, img, device=device)
        
        #print('is_bbox_over_threshold:',is_bbox_over_threshold(bbox,img))
        
        #if is_bbox_over_threshold(bbox,img)== False and mask_has_content(img,final_mask,threshold=50)==True:
        if mask_has_content(img,final_mask,threshold=50)==True:
            results.append( {
                #"bit_mask":bit_mask,
                "mask": final_mask,
                "group_size": len(group),   
            })
    #print(len(results))
    #out = sorted(results, key=lambda x: x['group_size'], reverse=True)
    return results



def is_bbox_over_threshold(bbox,image_rgb, threshold=2/3):
    #check if the bbox is over 2/3 of the size of image
    x1, y1,x2 ,y2 = bbox
    
    image_width, image_height = image_rgb.size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, image_width)
    y2 = min(y2, image_height)
    
    actual_w = x2 - x1
    actual_h = y2 - y1
    
    if actual_w <= 0 or actual_h <= 0:
        return False
    
    image_area = image_width * image_height
    bbox_area = actual_w * actual_h
    ratio = bbox_area / image_area
    #print('ratio:',ratio)
    return ratio > threshold


def mask_has_content(image, mask, threshold =20):
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    has_edges = np.any(masked_edges > 0)

    return has_edges
    

def warp_mask_with_flow(mask, flow,current_img,device):
    
    h, w = mask.shape[:2]

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = x + flow[..., 0]
    map_y = y + flow[..., 1]

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
    
    # wrap mask
    warped_mask = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # binary mask
    warped_mask = np.where(warped_mask > 0, 255, 0).astype(np.uint8)
    
    if warped_mask is None or warped_mask.sum() == 0:
        return None, None
    
    final_mask, _ = mask_post_processing(warped_mask, current_img, device=device)

    # 获取最大轮廓用于 bbox
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, x + w, y + h)
    else:
        return None,None
    
    return final_mask,bbox




def fuse_lk_farneback(ref_gray, curr_gray):
    h, w = ref_gray.shape

    # ---------------------- 步骤1: 全局运动补偿 ----------------------
    # 检测ORB特征点
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]  # 取最优100个匹配

    # 计算全局单应矩阵
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    curr_compensated = cv2.warpPerspective(curr_gray, H, (w, h))

    # calculate Farneback
    flow_farneback = cv2.calcOpticalFlowFarneback(
        prev=ref_gray,
        next=curr_compensated,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # ---------------------- 步骤3: 计算LK稀疏光流 ----------------------
    # 检测Shi-Tomasi角点
    corners = cv2.goodFeaturesToTrack(ref_gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
    p0 = np.float32(corners).reshape(-1, 1, 2)

    p1, status, _ = cv2.calcOpticalFlowPyrLK(ref_gray, curr_compensated, p0, None, winSize=(21, 21), maxLevel=5)
    valid_p0 = p0[status == 1].reshape(-1, 2)
    valid_p1 = p1[status == 1].reshape(-1, 2)
    flow_lk = valid_p1 - valid_p0  # 稀疏光流向量

    # 根据稀疏光流筛选运动物体区域（与全局模型不一致的点）
    motion_mask = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0), (dx, dy) in zip(valid_p0.astype(int), flow_lk):
        # 如果LK光流与Farneback预测差异大，标记为运动物体
        fb_dx = flow_farneback[y0, x0, 0]
        fb_dy = flow_farneback[y0, x0, 1]
        if np.linalg.norm([dx - fb_dx, dy - fb_dy]) > 1.0:  # 阈值根据场景调整
            cv2.circle(motion_mask, (x0, y0), 5, 255, -1)

    # 膨胀掩码，覆盖物体区域
    motion_mask = cv2.dilate(motion_mask, kernel=np.ones((15, 15), dtype=np.uint8))

    # ---------------------- 步骤4: 稀疏光流修正稠密光流 ----------------------
    # 在运动物体区域用LK光流覆盖Farneback结果
    for y in range(h):
        for x in range(w):
            if motion_mask[y, x] > 0:
                # 找到最近的LK光流点
                distances = np.linalg.norm(valid_p0 - [x, y], axis=1)
                nearest_idx = np.argmin(distances)
                if distances[nearest_idx] < 10:  # 最大搜索距离
                    flow_farneback[y, x] = flow_lk[nearest_idx]

    # ---------------------- 步骤5: 反向变换恢复原始坐标系光流 ----------------------
    # 由于之前补偿了镜头运动，需将光流还原到原始帧
    flow_final = cv2.warpPerspective(flow_farneback, H, (w, h), flags=cv2.WARP_INVERSE_MAP)

    return flow_final
   
    
def move_boxes_with_flow(flow, box, method='median', img=None):
    #boxes = [item['box'] for item in ref_info]
   
        
    x1, y1, x2, y2 = box
    
    # 将坐标转换为整数（确保在图像范围内）
    h, w = flow.shape[:2]
    x1, y1 = int(np.clip(x1, 0, w-1)), int(np.clip(y1, 0, h-1))
    x2, y2 = int(np.clip(x2, 0, w-1)), int(np.clip(y2, 0, h-1))
    
    # 检查框有效性
    if x1 >= x2 or y1 >= y2:
        return None
    
    # 提取框内光流
    flow_roi = flow[y1:y2, x1:x2]
    if flow_roi.size == 0:
        #moved_boxes.append([x1, y1, x2, y2])
        return None
    
    # 计算位移统计量
    if method == 'mean':
        dx = np.mean(flow_roi[..., 0])
        dy = np.mean(flow_roi[..., 1])
    elif method == 'median':
        dx = np.median(flow_roi[..., 0])
        dy = np.median(flow_roi[..., 1])
    else:
        raise ValueError("Method must be 'mean' or 'median'")
    
    # 调整框坐标
    new_box = [
        x1 + dx,
        y1 + dy,
        x2 + dx,
        y2 + dy
    ]
    
    # 限制坐标范围
    if img is not None:
        img_w,img_h = img.size
        new_box = [
            np.clip(new_box[0], 0, img_w-1),
            np.clip(new_box[1], 0, img_h-1),
            np.clip(new_box[2], 0, img_w-1),
            np.clip(new_box[3], 0, img_h-1)
        ]
    
    
    
    return new_box



def preprocess_image(img):
    """提升图像可跟踪性"""
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    # 边缘保留滤波
    return cv2.bilateralFilter(enhanced, 9, 75, 75)


def compensate_blur(img1, img2):
    """估计和补偿运动模糊"""
    kernel_size = 15
    # 估计点扩散函数
    psf = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    deblurred = cv2.filter2D(img2, -1, psf)
    return img1, deblurred

def _pyramid_motion_estimation(gray_ref_img, gray_curr_img, max_level=3):
    gray_ref_img = preprocess_image(gray_ref_img)
    gray_curr_img = preprocess_image(gray_curr_img)
    gray_ref_img, gray_curr_img = compensate_blur(gray_ref_img, gray_curr_img)

        # 计算当前层光流
    flow = cv2.calcOpticalFlowFarneback(
        prev=gray_ref_img, next=gray_curr_img,
        flow=None,
        pyr_scale=0.5,
        levels=5,  # 单层计算
        winsize=25,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
        
        
    h, w = gray_ref_img.shape
    step = max(1, w//100)  # 自适应步长
    src_pts = _generate_grid_points((h, w), step=step)
    flow_pts = flow[src_pts[:,1].astype(int), src_pts[:,0].astype(int)]
    dst_pts = src_pts + flow_pts
    M, _ = cv2.estimateAffine2D(
        src_pts.reshape(-1,1,2), 
        dst_pts.reshape(-1,1,2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    ) if len(src_pts) > 4 else (np.array([[1,0,0],[0,1,0]], dtype=np.float32), None)

   
    return M,flow

def _generate_grid_points(shape, step=10):
    """生成均匀网格点"""
    h, w = shape[:2]
    x = np.arange(0, w, step)
    y = np.arange(0, h, step)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _optimize_box(
    ref_box,
    M: np.ndarray,
    flow: np.ndarray,
    bit_mask: np.ndarray
):
    """基于运动场置信度的Box优化"""
    #print('####',M)

    h, w = flow.shape[:2]
    # 全局仿射变换
    pts = np.array([
        [ref_box[0], ref_box[1]],
        [ref_box[2], ref_box[1]],
        [ref_box[2], ref_box[3]],
        [ref_box[0], ref_box[3]]
    ], dtype=np.float32)
    global_pts = cv2.transform(pts.reshape(1,-1,2), M).squeeze()
    
    if bit_mask is not None:
        # 获取有效区域的运动向量
        mask_roi = bit_mask[
            max(0, ref_box[1]):min(h, ref_box[3]),
            max(0, ref_box[0]):min(w, ref_box[2])
        ]
        valid_flow = flow[
            max(0, ref_box[1]):min(h, ref_box[3]),
            max(0, ref_box[0]):min(w, ref_box[2])
        ][mask_roi > 0]
    else:
        valid_flow = flow[
            max(0, ref_box[1]):min(h, ref_box[3]),
            max(0, ref_box[0]):min(w, ref_box[2])
        ].reshape(-1, 2)
    
    # 4. 计算主要运动方向（改用中值更鲁棒）
    if len(valid_flow) > 10:
        dx, dy = np.median(valid_flow, axis=0)
    else:
        dx, dy = 0, 0
    
    # 5. 综合全局变换和局部运动
    adjusted_pts = global_pts + np.array([dx, dy])
    
    # 6. 计算新box坐标（保持原始宽高）
    width = ref_box[2] - ref_box[0]
    height = ref_box[3] - ref_box[1]
    center = np.mean(adjusted_pts, axis=0)
    
    new_box = [
        int(np.clip(center[0] - width/2, 0, w-1)),
        int(np.clip(center[1] - height/2, 0, h-1)),
        int(np.clip(center[0] + width/2, 0, w-1)),
        int(np.clip(center[1] + height/2, 0, h-1))
    ]
    
    
    return new_box



def _align_mask(
    ref_mask: np.ndarray,
    ref_img: np.ndarray,
    curr_img: np.ndarray,
    adjusted_box: List[int],
    M: np.ndarray
) -> np.ndarray:
    h, w = curr_img.shape[:2]
    
    # 基础仿射变换
    aligned_mask = cv2.warpAffine(ref_mask,M,(w, h),flags=cv2.INTER_LINEAR)
    # 相位
    try:
        shift, _ = cv2.phaseCorrelate(
            ref_img.astype(np.float32),
            curr_img.astype(np.float32)
        )
        M_shift = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
        aligned_mask = cv2.warpAffine(aligned_mask, M_shift, (w, h), flags=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"相位相关失败: {e}")

    # 应用边界约束
    x1, y1, x2, y2 = adjusted_box
    constraint = np.zeros_like(aligned_mask)
    constraint[y1:y2, x1:x2] = 1
    aligned_mask = (aligned_mask * constraint).astype(np.uint8)
    return aligned_mask



def process_frame(
        ref_info,
        curr_frame,
        ref_frame
    ):
        # Step 1: 金字塔运动估计
        M, flow = _pyramid_motion_estimation(
            ref_frame, curr_frame,
        )
        valid_info=[]
        for data in ref_info:
            # Step 2: Box优化
            #print('before:',data["box"])
            adjusted_box = _optimize_box(
                data["box"], M, flow,
                data["mask"]
            )
            #print('new::',adjusted_box)
            data["box"] = adjusted_box
            # Step 3: Mask对齐
            aligned_mask = _align_mask(
                data["mask"], ref_frame, curr_frame,
                adjusted_box, M
            )
            data["mask"]=aligned_mask
            valid_info.append(data)
        return valid_info


###############raft##################
def pil_to_tensor(pil_img):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    return transform(pil_img).unsqueeze(0)  # [1, 3, H, W]

def compute_flow(model, ref_pil, curr_pil):
    ref_tensor = pil_to_tensor(ref_pil)  # [1, 3, H, W]
    curr_tensor = pil_to_tensor(curr_pil)
    
    with torch.no_grad():
        flow = model(ref_tensor, curr_tensor)[0]  
    return flow.squeeze(0).permute(1, 2, 0).cpu().numpy()


def mask_crf_processing(patches_mask,bbox,image_rgb, device='cpu'):
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
