import importlib

import numpy as np
import torch

try:
    ext_module = importlib.import_module('geotransformer.ext')
except ImportError:
    ext_module = None


def grid_subsample(points, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    if ext_module is not None:
        s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
        return s_points, s_lengths
    return _grid_subsample_py(points, lengths, voxel_size)


def _grid_subsample_py(points, lengths, voxel_size):
    points_np = points.detach().cpu().numpy()
    lengths_np = lengths.detach().cpu().numpy()

    s_points_list = []
    s_lengths_list = []
    start_index = 0
    for length in lengths_np:
        length = int(length)
        cur_points = points_np[start_index : start_index + length]
        start_index += length

        if length == 0:
            cur_s_points = np.zeros((0, 3), dtype=np.float32)
        else:
            voxel_coords = np.floor(cur_points / voxel_size).astype(np.int64)
            _, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            num_voxels = int(inverse.max()) + 1

            sums = np.zeros((num_voxels, 3), dtype=np.float64)
            counts = np.zeros(num_voxels, dtype=np.int64)
            np.add.at(sums, inverse, cur_points)
            np.add.at(counts, inverse, 1)

            cur_s_points = (sums / counts[:, None]).astype(np.float32)

        s_points_list.append(cur_s_points)
        s_lengths_list.append(cur_s_points.shape[0])

    if s_points_list:
        s_points = torch.from_numpy(np.concatenate(s_points_list, axis=0))
    else:
        s_points = torch.zeros((0, 3), dtype=torch.float32)
    s_lengths = torch.tensor(s_lengths_list, dtype=torch.long)

    return s_points.to(points.device), s_lengths.to(lengths.device)
