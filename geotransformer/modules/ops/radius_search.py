import importlib

import numpy as np
import torch
from scipy.spatial import cKDTree

try:
    ext_module = importlib.import_module('geotransformer.ext')
except ImportError:
    ext_module = None


def radius_search(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function is implemented on CPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    if ext_module is not None:
        neighbor_indices = ext_module.radius_neighbors(
            q_points, s_points, q_lengths, s_lengths, radius
        )
    else:
        neighbor_indices = _radius_neighbors_py(
            q_points, s_points, q_lengths, s_lengths, radius
        )
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    return neighbor_indices


def _radius_neighbors_py(q_points, s_points, q_lengths, s_lengths, radius):
    q_points_np = q_points.detach().cpu().numpy()
    s_points_np = s_points.detach().cpu().numpy()
    q_lengths_np = q_lengths.detach().cpu().numpy()
    s_lengths_np = s_lengths.detach().cpu().numpy()

    total_s_points = int(s_points_np.shape[0])

    all_neighbors = []
    q_start_index = 0
    s_start_index = 0
    for q_length, s_length in zip(q_lengths_np, s_lengths_np):
        q_length = int(q_length)
        s_length = int(s_length)

        q_batch = q_points_np[q_start_index : q_start_index + q_length]
        s_batch = s_points_np[s_start_index : s_start_index + s_length]

        if q_length > 0:
            if s_length > 0:
                tree = cKDTree(s_batch)
                batch_neighbors = tree.query_ball_point(q_batch, r=radius)
                for qi, inds in enumerate(batch_neighbors):
                    if len(inds) > 1:
                        inds = np.asarray(inds)
                        diffs = s_batch[inds] - q_batch[qi]
                        order = np.argsort(np.einsum('ij,ij->i', diffs, diffs))
                        inds = inds[order]
                    all_neighbors.append([s_start_index + int(i) for i in inds])
            else:
                all_neighbors.extend([] for _ in range(q_length))

        q_start_index += q_length
        s_start_index += s_length

    num_queries = len(all_neighbors)
    max_count = max((len(x) for x in all_neighbors), default=0)

    neighbor_indices = np.full((num_queries, max_count), total_s_points, dtype=np.int64)
    for i, inds in enumerate(all_neighbors):
        if inds:
            neighbor_indices[i, : len(inds)] = inds

    return torch.from_numpy(neighbor_indices).to(q_points.device)
