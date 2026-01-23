import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import cv2

from scipy.spatial.distance import directed_hausdorff
import sys
sys.path.append("/netscratch/zlu/RAFT") 
sys.path.append("/netscratch/zlu/RAFT/core")
#from raft import RAFT
from core.raft import RAFT
from core.utils.utils import InputPadder

def compute_frame_affinity_matrix(features):
    '''
    Get the affinity matrix of one frame
    '''
    
    normalized_features = F.normalize(features, p=2, dim=-1)
    affinity_matrix = torch.matmul(normalized_features, normalized_features.transpose(-1, -2))
    affinity_matrix = torch.relu(affinity_matrix)
    return affinity_matrix


def Affinity_fusion(features_affinitymatrix, confidence_scores):
    target_affinity = features_affinitymatrix[:1].cuda() #[1, 1156, 1156]
    reference_affinities = features_affinitymatrix[1:].cuda() #[6, 1156, 1156]
    similarities = []
    for ref_affinity in reference_affinities:
        # Compute similarity using Frobenius inner product
        sim = torch.sum(target_affinity * ref_affinity) / (
            torch.norm(target_affinity, p='fro') * torch.norm(ref_affinity, p='fro') + 1e-8
        )
        similarities.append(sim)
    # Convert to tensor and normalize weights using softmax
    similarities = torch.tensor(similarities).cuda()  #  [6]
    
    weighted_similarities = similarities * confidence_scores[1:]

    weights = F.softmax(weighted_similarities, dim=0).cuda()  #  [6]
    # Step 2: Compute the weighted sum of reference affinities
    weighted_reference_affinity = torch.sum(
        weights[:, None, None] * reference_affinities, dim=0
    )  # [N, N]
    # Step 3: Fuse the target affinity with the weighted reference affinity
    enhanced_affinity = confidence_scores[0] * target_affinity + (1 - confidence_scores[0]) * weighted_reference_affinity

    return enhanced_affinity





     
def warp_features_with_flow(features, flow, size):
    """
    features: [B, C, H, W]  例如 [1, 384, 34, 34]
    flow: [B, 2, H, W]  光流 (x, y)
    size: target spatial size (H, W), e.g., (34, 34)
    """
    B, C, H, W = features.size()

    # Create mesh grid
    y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack((x, y), 2).float().to(features.device)  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

    # Add flow
    flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    grid = grid + flow

    # Normalize grid to [-1, 1]
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

    # Sample features using grid_sample
    warped_features = F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)

    return warped_features
