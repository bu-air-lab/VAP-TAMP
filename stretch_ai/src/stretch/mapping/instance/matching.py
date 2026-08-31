# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from stretch.utils.bboxes_3d import (
    box3d_intersection_from_bounds,
    box3d_overlap_from_bounds,
    box3d_volume_from_bounds,
)


class Bbox3dOverlapMethodEnum(Enum):
    IOU = "IOU"
    ONE_SIDED_IOU = "ONE_SIDED_IOU"


@dataclass
class ViewMatchingConfig:
    within_class: bool = True

    box_match_mode: Bbox3dOverlapMethodEnum = Bbox3dOverlapMethodEnum.ONE_SIDED_IOU
    box_overlap_eps: float = 1e-7
    box_min_iou_thresh: float = 0.02  # Very permissive for cross-viewpoint matching
    box_overlap_weight: float = 0.2   # Less emphasis on exact bbox match

    visual_similarity_weight: float = 0.5  # Balanced visual features
    shape_similarity_weight: float = 0.3   # New: Shape-based matching
    min_similarity_thresh: float = 0.25    # Lower threshold for matches
    
    # Shape matching parameters
    use_shape_matching: bool = True
    shape_tolerance: float = 0.3           # Tolerance for shape differences


def get_bbox_similarity(
    bounds1: Union[Tensor, List[Tensor]],
    bounds2: Union[Tensor, List[Tensor]],
    overlap_eps: float = 1e-6,
    mode: Bbox3dOverlapMethodEnum = Bbox3dOverlapMethodEnum.ONE_SIDED_IOU,
) -> Tensor:
    if len(bounds1) == 0:
        return None
    if len(bounds2) == 0:
        return None
    if not isinstance(bounds1, Tensor):
        bounds1 = torch.stack(bounds1, dim=0)
    if not isinstance(bounds2, Tensor):
        bounds2 = torch.stack(bounds2, dim=0)

    if mode == Bbox3dOverlapMethodEnum.ONE_SIDED_IOU:
        volume1 = box3d_volume_from_bounds(bounds1)
        assert torch.all(volume1 > 0.0), bounds1
        vol_int, _ = box3d_overlap_from_bounds(bounds1, bounds2, overlap_eps)
        ious = vol_int / volume1.unsqueeze(1)
    elif mode == Bbox3dOverlapMethodEnum.IOU:
        _, ious = box3d_overlap_from_bounds(bounds1, bounds2, overlap_eps)
    else:
        raise NotImplementedError(f"Unsupported Bbox3dOverlapMethodEnum mode: {mode}")
    assert ious.ndim == 2 and ious.shape[0] == len(bounds1), ious.shape
    return ious


class EncoderSimilarityMethodEnum(Enum):
    MAX = auto()
    # MEAN = auto()


def dot_product_similarity(feats1, feats2, normalize=True):
    """
    Calculate the cosine similarity between two sets of feature vectors.

    Args:
        feats1: NxD tensor (N: number of vectors, D: dimensionality of each vector)
        feats2: MxD tensor (M: number of vectors, D: dimensionality of each vector)
        normalize: Whether to normalize the input feature vectors. Default is True.

    Returns:
        N x M tensor of similarities
    """
    if feats1 is None or len(feats1) == 0:
        return None
    if feats2 is None or len(feats2) == 0:
        return None
    if not isinstance(feats1, Tensor):
        feats1 = torch.stack(feats1, dim=0)
    if not isinstance(feats2, Tensor):
        feats2 = torch.stack(feats2, dim=0)
    if normalize:
        # Normalize the input feature vectors to have unit L2 norm
        feats1 = feats1 / torch.norm(feats1, dim=1, keepdim=True)
        feats2 = feats2 / torch.norm(feats2, dim=1, keepdim=True)

    # Calculate the dot product between the (optionally) normalized feature vectors
    dot_product = torch.mm(feats1, feats2.t())

    return dot_product


def get_shape_similarity(
    bounds1: Union[Tensor, List[Tensor]],
    bounds2: Union[Tensor, List[Tensor]],
    tolerance: float = 0.3,
) -> Tensor:
    """
    Calculate shape similarity based on bounding box dimensions and aspect ratios.
    This helps recognize the same object from different viewpoints.
    
    Args:
        bounds1: First set of bounding boxes [N, 3, 2]
        bounds2: Second set of bounding boxes [M, 3, 2] 
        tolerance: Tolerance for shape differences (0.0=exact, 1.0=very loose)
        
    Returns:
        N x M tensor of shape similarities [0, 1]
    """
    if len(bounds1) == 0 or len(bounds2) == 0:
        return None
        
    if not isinstance(bounds1, Tensor):
        bounds1 = torch.stack(bounds1, dim=0)
    if not isinstance(bounds2, Tensor):
        bounds2 = torch.stack(bounds2, dim=0)
    
    # Calculate dimensions for each bounding box
    dims1 = bounds1[:, :, 1] - bounds1[:, :, 0]  # [N, 3] (width, height, depth)
    dims2 = bounds2[:, :, 1] - bounds2[:, :, 0]  # [M, 3]
    
    # Calculate volumes
    vols1 = torch.prod(dims1, dim=1, keepdim=True)  # [N, 1]
    vols2 = torch.prod(dims2, dim=1, keepdim=True)  # [M, 1]
    
    # Calculate aspect ratios (width/height, width/depth, height/depth)
    ratios1 = torch.stack([
        dims1[:, 0] / (dims1[:, 1] + 1e-6),  # width/height
        dims1[:, 0] / (dims1[:, 2] + 1e-6),  # width/depth  
        dims1[:, 1] / (dims1[:, 2] + 1e-6),  # height/depth
    ], dim=1)  # [N, 3]
    
    ratios2 = torch.stack([
        dims2[:, 0] / (dims2[:, 1] + 1e-6),
        dims2[:, 0] / (dims2[:, 2] + 1e-6),
        dims2[:, 1] / (dims2[:, 2] + 1e-6),
    ], dim=1)  # [M, 3]
    
    # Compare volumes (normalized by larger volume)
    vol_max = torch.maximum(vols1, vols2.t())  # [N, M]
    vol_min = torch.minimum(vols1, vols2.t())  # [N, M]
    vol_similarity = vol_min / (vol_max + 1e-6)  # [N, M]
    
    # Compare aspect ratios using cosine similarity
    ratios1_norm = ratios1 / (torch.norm(ratios1, dim=1, keepdim=True) + 1e-6)
    ratios2_norm = ratios2 / (torch.norm(ratios2, dim=1, keepdim=True) + 1e-6)
    ratio_similarity = torch.mm(ratios1_norm, ratios2_norm.t())  # [N, M]
    
    # Combine volume and ratio similarities
    shape_similarity = 0.6 * vol_similarity + 0.4 * torch.clamp(ratio_similarity, 0, 1)
    
    # Apply tolerance - higher tolerance makes more things similar
    shape_similarity = torch.pow(shape_similarity, 1.0 - tolerance)
    
    return torch.clamp(shape_similarity, 0, 1)


# Geometry-based matching
def find_global_instance_by_pointcloud_overlap(
    self, env_id: int, local_instance_view_id: int
) -> Optional[int]:
    """
    Find the global instance ID that has the most overlapping points in the point cloud
    with the local instance identified by `local_instance_id` in the environment specified by `env_id`.

    This function performs the following steps to associate the local instance with a global instance:
    1. Compute the 3D box intersection between the local instance's 3D bounding box and those of all global instances.
    2. Filter both local and global instance point clouds by this intersection.
    3. Compute the distance to the nearest global points from the local instance's point cloud.
            nearest_global_point = knn(instance_view.points_filtered, global_instance.points_filtered)
    4. Determine the percentage of points in the local instance's filtered point cloud that are near to points in the global instances' point clouds.
            points_matched = % of instance_view.points_filtered[nearest_point_dist < dist_thresh]
    5. Associate the local instance with the global instance based on one of the following metrics:
        - The (% matched points) * one-sided IoU: one_sided_IoU * points_matched.mean()
        - The sum of matched points * points_matched.sum()

    Args:
        env_id (int): The environment ID.
        local_instance_view_id (int): The local instance view ID whose global counterpart needs to be found.

    Returns:
        Optional[int]: The global instance ID with the most point cloud overlap.
                    Returns None if no such global instance is found.

    TODO:
        - Optimize by having global instances store a voxelized point cloud to keep the number of points manageable.
    """
    raise NotImplementedError(
        "Placeholder pending correct implementation of geometry based matching"
    )
    # get instance view
    instance_view = self.get_local_instance_view(env_id, local_instance_view_id)
    volume1 = box3d_volume_from_bounds(instance_view.bounds)

    if instance_view is not None:
        global_instance_ids = self.get_global_instance_ids(env_id)
        if len(global_instance_ids) == 0:
            return None
        instances = self.get_instances_by_ids(env_id, global_instance_ids)
        global_bounds = torch.stack([inst.bounds for inst in instances], dim=0)
        vol_int, iou, intersection_bounds = box3d_intersection_from_bounds(
            instance_view.bounds.unsqueeze(0), global_bounds, self.overlap_eps
        )
        # 2. Filter by intersection_bounds
        # 3. nearest_global_point = knn(instance_view.points_filtered, global_instance.points_filtered)
        # 4. points_matched = % of instance_view.points_filtered[nearest_point_dist < dist_thresh]
        # 5.
        ious = vol_int / volume1
        assert ious.ndim == 2 and ious.shape[0] == 1, ious.shape
        ious = ious.flatten()

        if ious.max() > self.iou_threshold:
            return global_instance_ids[ious.argmax()]
    return None


def get_similarity(
    instance_bounds1: Tensor,
    instance_bounds2: Tensor,
    visual_embedding1: Tensor,
    visual_embedding2: Tensor,
    text_embedding1: Optional[Tensor] = None,
    text_embedding2: Optional[Tensor] = None,
    view_matching_config: ViewMatchingConfig = ViewMatchingConfig(),
    verbose: bool = False,
):
    """Compute similarity based on bounding boxes, visual features, and shape"""
    # BBox similarity
    overlap_similarity = get_bbox_similarity(
        instance_bounds1,
        instance_bounds2,
        overlap_eps=view_matching_config.box_overlap_eps,
        mode=view_matching_config.box_match_mode,
    )
    if verbose:
        print(f"geometric similarity score: {overlap_similarity}")
    similarity = overlap_similarity * view_matching_config.box_overlap_weight

    # Visual similarity
    if view_matching_config.visual_similarity_weight > 0.0:
        visual_similarity = nn.CosineSimilarity(dim=1)(
            visual_embedding1, torch.stack(visual_embedding2, dim=0)
        ).unsqueeze(0)
        if verbose:
            print(f"visual similarity score: {visual_similarity}")
        # Handle the case where there is no embedding to examine
        # If we return visual similarity, only then do we use it
        if visual_similarity is not None:
            visual_similarity[overlap_similarity < view_matching_config.box_min_iou_thresh] = 0.0
            similarity += visual_similarity * view_matching_config.visual_similarity_weight

    # Shape similarity - NEW: This helps with cross-viewpoint recognition
    if hasattr(view_matching_config, 'use_shape_matching') and view_matching_config.use_shape_matching and hasattr(view_matching_config, 'shape_similarity_weight') and view_matching_config.shape_similarity_weight > 0.0:
        shape_similarity = get_shape_similarity(
            instance_bounds1,
            instance_bounds2,
            tolerance=getattr(view_matching_config, 'shape_tolerance', 0.3)
        )
        if verbose and shape_similarity is not None:
            print(f"shape similarity score: {shape_similarity}")
        if shape_similarity is not None:
            similarity += shape_similarity * view_matching_config.shape_similarity_weight

    return similarity
