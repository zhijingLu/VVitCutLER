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


def cluster_videomask_by_iou(masks,framelist ,threshold=0.5, pivot_iter=5, device='cpu'):
    """
    This method performs clustering of masks based on the IoU between them. It is greedy algorithm that tries to find
    the largest clusters of masks by IoU distance between them. The return value is a list of clusters dictionaries
    that contain the indices of the masks that belong to the cluster and other information about the cluster.
    :param masks: numpy array of shape (N, H, W) where N is the number of masks
    :param threshold: IoU threshold. Above this threshold masks are considered to be in the same cluster.
    :param pivot_iter: number of iterations to find the pivot mask. The pivot mask is the mask that has the most
    number of masks close masks above the threshold.
    """
    '''
    ious = iou_between_masks(masks, device=device) 
    framelist = np.array(framelist)
    current_frame_indices =  np.where(framelist == 0)[0]
    ref_frame_indices = np.where(framelist == 1)[0] # Masks from frame 1
    selected_mask = []
    used_indices = set()
    for pivot_index in current_frame_indices:
        above_threshold = np.array(
            [idx for idx in ref_frame_indices if idx not in used_indices and ious[pivot_index, idx] > threshold]
        )
        above_threshold = np.concatenate(([pivot_index], above_threshold))
        above_threshold = above_threshold.astype(int)
        #print('above_threshold:',above_threshold)
        # Store cluster information
        pivot_mask = masks[pivot_index]
        pivot_bbox = bbox_from_mask(pivot_mask)
        selected_mask.append({
            'pivot_index': pivot_index,
            'pivot_mask': pivot_mask,
            'pivot_bbox': pivot_bbox,
            'segment_cluster_indices': above_threshold,
            'cluster_size': len(above_threshold),  # Include the pivot mask itself
        })

        # Mark these frame 1 masks as used
        used_indices.update(above_threshold)
    '''
    #print(selected_mask)
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
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    patches_mask = F.interpolate(mask_tensor, size=rescale_size, mode='nearest')[0, 0].cpu().numpy()
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


def iou_clustering(masks_proposals,
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
    #
    # create numpy array from the masks list of numpy arrays
    masks = np.array(masks_proposals)
    # cluster the masks by IoU
    mask_clusters = cluster_mask_by_iou(masks, threshold=0.6, device=device)
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
        final_mask, success = mask_post_processing(bit_mask, image_rgb, device='cpu')
        bbox = bbox_from_mask(final_mask)
        bbox = (bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])
        #print('getmask bbox',bbox)
        image_masks.append({
            "bit_mask": bit_mask,
            "mask": final_mask,
            "box":bbox,
            "crf_success": success,
            "cluster_size": cluster_size,
            "background":False,
            "weight":1,
            "is_current":True
        })

    out = sorted(image_masks, key=lambda x: x['cluster_size'], reverse=True)
    return out




def clustering(masks_proposals,
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



def getMasks(image_rgb, eig_vec_groups, Ks=(2, 3), tau_m=0.2, device='cpu'):

    kmeans_labels = kmeans_labeling(eig_vec_groups, Ks=Ks)
    masks_proposals = instances_from_semantic_labels(kmeans_labeling_list(kmeans_labels))
    image_masks = iou_clustering(masks_proposals, image_rgb, tau_m=tau_m, device=device)
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
    #print('into ',len(data_list))
    groups = []
    visited = set()
    for i in range(len(data_list)):
        #if "box" not in data_list[i]:
        #    continue 
        if i in visited:
            continue
        current_group = [data_list[i]]
        visited.add(i)
        for j in range(i+1, len(data_list)):
            #if "box" not in data_list[j]:
            #    continue 
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
    '''
    for i, group in enumerate(groups):
        print(f"Group {i+1}: {len(group)} boxes")
    '''
    return groups

#def process_mask_group(mask_group,img, tau_m=0.5,device='cpu'):
    
def group_by_mask_iou(data_list, iou_threshold=0.6, max_masks_per_img=10):

    groups = []
    visited = set()

    for i in range(len(data_list)):
        if i in visited:
            continue
        current_group = [data_list[i]]
        visited.add(i)

        for j in range(i + 1, len(data_list)):
            if j in visited:
                continue

            iou = compute_mask_iou(data_list[i]["mask"], data_list[j]["mask"])
            if iou > iou_threshold:
                current_group.append(data_list[j])
                visited.add(j)

        groups.append(current_group)

    groups = sorted(groups, key=lambda g: len(g), reverse=True)
    groups = groups[:max_masks_per_img]
    '''
    for i, group in enumerate(groups):
        print(f"Group {i+1}: {len(group)} masks")
    '''
    return groups

def group_by_mask_iou_2(cluster_masks, iou_threshold=0.6):

    clusters = []
    for mask in cluster_masks:
        placed = False
        for cluster in clusters:
            # Check if mask is similar to any mask in the cluster
            if any(compute_mask_iou(mask, cmask) > iou_threshold for cmask in cluster):
                cluster.append(mask)
                placed = True
                break
        if not placed:
            clusters.append([mask])
    return clusters


def video_mask_grouping_cluster_nobg(data_list, img,iou_thresh=0.6, tau_m=0.5,device='cpu'):
    groups = group_by_iou(data_list, iou_thresh)
    
    results = []
    for group in groups:
        if len(group)>1:
            #print(len(group))
            cluster_masks = np.array([item['mask'] for item in group])  # shape: (N, H, W)
            cluster_background = [item['background'] for item in group]
            true_count = sum(cluster_background)
            #print('true_count',true_count)
            if true_count>=2:
                continue
            
            minigroup=group_by_mask_iou_2(cluster_masks, iou_threshold=0.5)
            #print(len(minigroup))
            for mini in minigroup:
                mini=np.array(mini)
                mask = np.sum(mini, axis=0)/mini.shape[0]
            
                #tau_m = np.mean(mask) + np.std(mask)
            #mask = (mask > threshold).astype(np.uint8) 
            
                mask = (mask > tau_m).astype(np.uint8) 

                final_mask, success = mask_post_processing(mask, img, device=device)
                if is_valid_shape(final_mask) and detect_jaggedness(final_mask):
                    results.append( {
                        #"bit_mask":bit_mask,
                    "mask": final_mask,
                    "group_size": len(group),   
                    })
        else:
            if group[0]['background']==False:
                final_mask = group[0]['mask']
                if is_valid_shape(final_mask) and detect_jaggedness(final_mask):
                    results.append( {
                    #"bit_mask":bit_mask,
                        "mask": final_mask,
                        "group_size": len(group),   
                        })
            
        # and is_background(img,final_mask)== False :
        

    return results

def video_mask_grouping_cluster(data_list, img,iou_thresh=0.6, tau_m=0.5,device='cpu'):
    groups = group_by_iou(data_list, iou_thresh)
    
    results = []
    for group in groups:
        
        cluster_masks = np.array([item['mask'] for item in group])  # shape: (N, H, W)
        cluster_weight = np.array([item['weight'] for item in group])
        cluster_weight = cluster_weight[:, None, None]  
        weighted_mask = np.sum(cluster_masks * cluster_weight, axis=0) / (np.sum(cluster_weight) + 1e-6)

        #tau_m = np.mean(weighted_mask) + np.std(weighted_mask)
        #mask = (mask > threshold).astype(np.uint8) 
        mask = (weighted_mask > tau_m).astype(np.uint8) 

        final_mask, success = mask_post_processing(mask, img, device=device)
        if is_valid_shape(final_mask) and detect_jaggedness(final_mask):
            results.append( {
                        #"bit_mask":bit_mask,
                    "mask": final_mask,
                    "group_size": len(group),   
                })

    return results

def compute_mask_iou(mask1, mask2):
   
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0
    else:
        return intersection / union

def filter_current_masks_by_ref(current_info, ref_info, iou_thresh=0.8):
    
    filtered_info = []
    current_masks = [item['mask'] for item in current_info]
    #print('current_masks',len(current_masks))
    ref_masks = [refitem['mask'] for refitem in ref_info]
    #print('ref:',len(ref_masks))
    for idx, cur_mask in enumerate(current_masks):
        max_iou = 0
        for ref_mask in ref_masks:
            iou = compute_mask_iou(cur_mask, ref_mask)
            if iou > max_iou:
                max_iou = iou
        if max_iou >= iou_thresh:
            filtered_info.append(current_info[idx]) 
    #print(len(filtered_masks))
    
    return filtered_info

def filter_current_box_by_ref(current_info, ref_info, iou_thresh=0.8):
    
    filtered_info = []
    current_masks = [item['box'] for item in current_info]
    #print('current_masks',current_masks)
    ref_masks = [refitem['box'] for refitem in ref_info]
    #print('ref:',ref_masks)
    for idx, cur_mask in enumerate(current_masks):
        max_iou = 0
        for ref_mask in ref_masks:
            iou = compute_iou(cur_mask, ref_mask)
            if iou > max_iou:
                max_iou = iou
        if max_iou >= iou_thresh:
            filtered_info.append(current_info[idx]) 
    #print('+++',len(filtered_info))
    
    return filtered_info


def is_valid_shape(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    if area < 5000 or aspect_ratio > 5 or aspect_ratio < 0.2:
        return False
    return True

def is_background(pil_img, mask, color_thresh=10.0, texture_thresh=5.0):
    img_rgb = np.array(pil_img)
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Input image must be RGB")

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if img_bgr.shape[:2] != mask.shape:
        raise ValueError("Image and mask size mismatch")

    masked_pixels = img_bgr[mask > 0]
    if len(masked_pixels) == 0:
        return True  

    std_rgb = np.std(masked_pixels, axis=0)
    mean_std_rgb = np.mean(std_rgb)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    masked_lap = lap[mask > 0]
    var_lap = np.var(masked_lap)

    if mean_std_rgb < color_thresh and var_lap < texture_thresh:
        return True
    return False


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arccos(np.clip(np.dot(v1, v2) / 
                              (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1.0, 1.0))
    return np.degrees(angle)



def detect_jaggedness(mask):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)
    # 2. 轮廓提取
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    jagged_score_list = []
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        if len(cnt) < 5:  # 小轮廓跳过
            continue

        angles = []
        for i in range(1, len(cnt)-1):
            p1, p2, p3 = cnt[i-1], cnt[i], cnt[i+1]
            angle = calculate_angle(p1, p2, p3)
            angles.append(angle)

        angles = np.array(angles)
        if len(angles) == 0:
            return True

        sharp_corners = np.sum(angles < 160)
        jagged_score = sharp_corners / len(angles)
        jagged_score_list.append(jagged_score)

    if jagged_score_list:
        final_score = np.mean(jagged_score_list)
        #print(f"锯齿感评分: {final_score:.3f}")
        
        if final_score > 0.2:
            return True
        else:
            return False
    else:
        return False



def video_grouping_refine(data_list, img,iou_thresh=0.6, tau_m=0.5,device='cpu'):
    groups = group_by_iou(data_list, iou_thresh)
    
    results = []
    for group in groups:
        if len(group)>1:
            #print(len(group))
            cluster_masks = np.array([item['mask'] for item in group])  # shape: (N, H, W)
            cluster_background = [item['background'] for item in group]
            cluster_current = [item['is_current'] for item in group]
            #print('cluster_current',cluster_current)
            true_bgcount = sum(cluster_background)
            has_current_frame = sum(cluster_current)
            #print(has_current_frame)
            #print('true_count',true_count)
            if true_bgcount>=2 or has_current_frame==0:
                continue
            current_masks = cluster_masks[cluster_current]
            #print(len(current_masks))
            mask = np.sum(current_masks, axis=0)/current_masks.shape[0]
            
                #tau_m = np.mean(mask) + np.std(mask)
            #mask = (mask > threshold).astype(np.uint8) 
            
            mask = (mask > tau_m).astype(np.uint8) 

            final_mask, success = mask_post_processing(mask, img, device=device)
            if is_valid_shape(final_mask) and detect_jaggedness(final_mask):
                results.append( {
                    #"bit_mask":bit_mask,
                "mask": final_mask,
                "group_size": len(group),   
                })
        else:
            if group[0]['background']==False and group[0]['is_current']==True:
                final_mask = group[0]['mask']
                if is_valid_shape(final_mask) and detect_jaggedness(final_mask):
                    results.append( {
                    #"bit_mask":bit_mask,
                        "mask": final_mask,
                        "group_size": len(group),   
                        })
            
        # and is_background(img,final_mask)== False :
        

    return results