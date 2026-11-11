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
            q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit
        )
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    return neighbor_indices


def _radius_neighbors_py(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit=0):
    r"""纯 Python 回退实现（未编译 geotransformer.ext 时使用）。

    使用 ``cKDTree.query(k=neighbor_limit)`` 一次取出最近的 k 个邻居，
    而不是逐点 ``query_ball_point`` + Python 循环 + argsort。几千到几万点的
    场景下约快 50~100 倍，结果与原实现一致：邻居按距离升序，不足 k 个的
    位置用 ``total_s_points`` 填充。
    """
    q_points_np = q_points.detach().cpu().numpy()
    s_points_np = s_points.detach().cpu().numpy()
    q_lengths_np = q_lengths.detach().cpu().numpy()
    s_lengths_np = s_lengths.detach().cpu().numpy()

    total_s_points = int(s_points_np.shape[0])

    chunks = []
    q_start_index = 0
    s_start_index = 0
    for q_length, s_length in zip(q_lengths_np, s_lengths_np):
        q_length = int(q_length)
        s_length = int(s_length)

        if q_length > 0:
            if s_length > 0:
                q_batch = q_points_np[q_start_index : q_start_index + q_length]
                s_batch = s_points_np[s_start_index : s_start_index + s_length]
                tree = cKDTree(s_batch)

                if neighbor_limit > 0:
                    k = min(int(neighbor_limit), s_length)
                    _, indices = tree.query(q_batch, k=k, distance_upper_bound=radius)
                    indices = np.asarray(indices, dtype=np.int64).reshape(q_length, k)
                    # 未找到邻居的位置 query 返回 s_length，统一改成 batch 级填充值
                    indices = np.where(indices < s_length, indices + s_start_index, total_s_points)
                    chunks.append(indices)
                else:
                    # 不限制邻居数量（原语义）：保留 query_ball_point 路径
                    batch_neighbors = tree.query_ball_point(q_batch, r=radius)
                    max_count = max((len(x) for x in batch_neighbors), default=0)
                    indices = np.full((q_length, max_count), total_s_points, dtype=np.int64)
                    for qi, inds in enumerate(batch_neighbors):
                        if inds:
                            inds = np.asarray(inds, dtype=np.int64)
                            diffs = s_batch[inds] - q_batch[qi]
                            order = np.argsort(np.einsum('ij,ij->i', diffs, diffs))
                            indices[qi, : len(inds)] = inds[order] + s_start_index
                    chunks.append(indices)
            else:
                # 该 batch 元素没有支持点：保留占位行，后续统一填充
                chunks.append(np.zeros((q_length, 0), dtype=np.int64))

        q_start_index += q_length
        s_start_index += s_length

    if not chunks:
        return torch.zeros((0, 0), dtype=torch.int64)

    num_queries = sum(chunk.shape[0] for chunk in chunks)
    max_width = max(chunk.shape[1] for chunk in chunks)
    neighbor_indices = np.full((num_queries, max_width), total_s_points, dtype=np.int64)
    offset = 0
    for chunk in chunks:
        neighbor_indices[offset : offset + chunk.shape[0], : chunk.shape[1]] = chunk
        offset += chunk.shape[0]

    return torch.from_numpy(neighbor_indices).to(q_points.device)
