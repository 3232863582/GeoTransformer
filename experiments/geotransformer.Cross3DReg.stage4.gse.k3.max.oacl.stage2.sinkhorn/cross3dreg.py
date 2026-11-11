r"""Cross3DReg 数据集适配加载器。

数据目录结构::

    <dataset_root>/
    |-- metadata/
    |   |-- train.pkl
    |   |-- val.pkl
    |   `-- test.pkl
    `-- sequences/
        `-- <seq_id>/                 # '00' ~ '10'
            |-- array/<frame>.ply     # 12000 点
            |-- lidar/<frame>.ply     # 32768 点
            `-- image/<frame>.jpg

metadata 是 joblib 序列化的 ``list[dict]``，每条样本包含：

===============  =====================================================
键               含义
===============  =====================================================
``frame0/1``     帧号字符串（如 '000000'），本数据集中二者相同
``pcd0/pcd1``    相对 dataset_root 的 ply 路径（array / lidar）
``seq_id``       场景编号（'00' ~ '10'）
``image``        对应相机图像路径（本加载器不使用）
``p2`` / ``tr``  相机投影矩阵 / 相机-激光外参（本加载器不使用）
``transform``    4x4 真值变换
===============  =====================================================

变换方向约定（已在本数据集上实测验证）：
    metadata['transform'] 把 **pcd0(array) -> pcd1(lidar)**，即 array -> lidar；
    而 GeoTransformer 全流程要求 ``data_dict['transform']`` 把 **src -> ref**。
    本数据集 lidar 各帧已配准到同一全局坐标系（帧间最近邻距离约 8mm），
    array 各帧各自漂移，因此取 ``ref = pcd1``（lidar，全局基准）、
    ``src = pcd0``（array，待配准），transform 直接取 metadata 原值，无需求逆：:

        ref_points = src_points @ rotation.T + translation

本数据集 metadata 没有 ``overlap`` 字段：
    - 不做 overlap 过滤（除非自行补充该字段）
    - ``data_dict['overlap']`` 默认填 1.0；``compute_overlap=True`` 时用 KDTree 估计

用法::

    from cross3dreg import Cross3DRegDataset

    dataset = Cross3DRegDataset('D:/ourdata/Cross3DReg', 'train', point_limit=30000, use_augmentation=True)
    sample = dataset[0]
    print(sample['ref_points'].shape, sample['src_points'].shape, sample['transform'].shape)

在 ``dataset.py`` 中替换现有 ``crossdata``（注意现有 crossdata.py 里基类名
``torch.utils.data.Datset`` 拼写有误，会直接报错）::

    from cross3dreg import Cross3DRegDataset as crossdata
"""

import os.path as osp
import pickle
import random
import warnings
from typing import Dict
import open3d as o3d
import numpy as np
import torch
import torch.utils.data

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)

DEFAULT_DATASET_ROOT = r'/media/lxy/新加卷1/zj/Cross3DReg'
# DEFAULT_DATASET_ROOT = r'D:\zj\Cross3DReg'

# Cross3DReg 只有 train / val / test 划分；评估脚本里的 benchmark 名称
# 一律映射到 test（本数据集没有 overlap 字段，无法区分 3DMatch/3DLoMatch）


_PLY_TYPES = {
    'char': 'i1', 'int8': 'i1',
    'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2',
    'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4',
    'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4',
    'double': 'f8', 'float64': 'f8',
}


def _read_ply_xyz_numpy(file_path):
    """不依赖第三方库的 PLY 读取（仅 vertex 的 x/y/z，支持 binary/ascii）。"""
    with open(file_path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'不是合法的 PLY 文件: {file_path}')

        fmt = None
        elements = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'PLY 头部不完整: {file_path}')
            text = line.decode('ascii', 'ignore').strip()
            if text == 'end_header':
                break
            if not text or text.startswith('comment') or text.startswith('obj_info'):
                continue
            tokens = text.split()
            if tokens[0] == 'format':
                fmt = tokens[1]
            elif tokens[0] == 'element':
                elements.append({'name': tokens[1], 'count': int(tokens[2]), 'props': []})
            elif tokens[0] == 'property':
                if tokens[1] == 'list':
                    elements[-1]['props'].append(None)
                else:
                    elements[-1]['props'].append((tokens[2], _PLY_TYPES[tokens[1]]))

        vertex = next((e for e in elements if e['name'] == 'vertex'), None)
        if vertex is None:
            raise ValueError(f'PLY 中没有 vertex 元素: {file_path}')
        if any(prop is None for prop in vertex['props']):
            raise ValueError(f'不支持带 list 属性的 vertex: {file_path}')

        names = [name for name, _ in vertex['props']]
        sel = [names.index(k) for k in ('x', 'y', 'z')]

        if fmt in ('binary_little_endian', 'binary_big_endian'):
            endian = '<' if fmt == 'binary_little_endian' else '>'
            count = vertex['count']
            codes = [code for _, code in vertex['props']]
            if names == ['x', 'y', 'z'] and len(set(codes)) == 1:
                # 常见情况（Open3D 导出，只有 x/y/z）：一次 frombuffer 再 reshape，
                # 比构造 structured dtype 再 np.stack 三列快约 2 倍。
                base_dtype = np.dtype(endian + codes[0])
                data = np.frombuffer(f.read(base_dtype.itemsize * count * 3), dtype=base_dtype, count=count * 3)
                return data.reshape(count, 3).astype(np.float64)
            dtype = np.dtype([(name, endian + code) for name, code in vertex['props']])
            data = np.frombuffer(f.read(dtype.itemsize * vertex['count']), dtype=dtype, count=vertex['count'])
            return np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)

        if fmt == 'ascii':
            values = np.array(f.read().decode('ascii', 'ignore').split(), dtype=np.float64)
            values = values[: vertex['count'] * len(names)].reshape(-1, len(names))
            return values[:, sel]

        raise ValueError(f'不支持的 PLY 格式: {fmt}')


def read_ply_points(file_path):
    """读取 PLY 点云，返回 (N, 3) float64 数组。

    直接按二进制头部偏移解析，不再经过 plyfile（本数据集只有 x/y/z 三个
    double 属性，实测快约 2 倍，也少一次 Python 层属性解析）。
    """
    return _read_ply_xyz_numpy(file_path)

def voxel_downsample_to_target(pcd_path, target_num_points=30000, initial_voxel=0.1, max_iter=25):
        pcd = o3d.io.read_point_cloud(pcd_path)
        # 指数探测：确定包含目标点数的体素区间 [lower, upper]
        voxel_size = initial_voxel
        num_points = len(pcd.voxel_down_sample(voxel_size).points)

        if num_points < target_num_points:
            lower = 0.0
            upper = voxel_size
        else:
            lower = voxel_size
            upper = voxel_size
            while num_points > target_num_points:
                upper *= 2
                num_points = len(pcd.voxel_down_sample(upper).points)

        # 二分逼近 target ± 5%
        best = pcd.voxel_down_sample(upper)
        for _ in range(max_iter):
            voxel_size = (lower + upper) / 2
            down_pcd = pcd.voxel_down_sample(voxel_size)
            num_points = len(down_pcd.points)

            if abs(num_points - target_num_points) < target_num_points * 0.05:
                return np.asarray(down_pcd.points)  # 控制误差在±5%内

            if num_points > target_num_points:
                lower = voxel_size
            else:
                upper = voxel_size

            best = down_pcd
        return np.asarray(best.points)  # 最终返回最后一次采样结果

def load_metadata(file_path):
    """读取 metadata .pkl（joblib 或 pickle 序列化的文件均可）。"""
    try:
        import joblib
        return joblib.load(file_path)
    except ImportError:
        pass
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def invert_transform(rotation, translation):
    """把 (R, t) 求逆：p' = R p + t  <->  p = R^T p' - R^T t。"""
    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation
    return rotation_inv, translation_inv


def find_correspondences(ref_points, src_points, transform, matching_radius):
    """返回真值对应点索引 (C, 2)：ref 点在变换后的 src 的半径邻域内。

    与 geotransformer.utils.registration.get_correspondences 等价，
    但使用 int64 而非已被 NumPy 2.x 移除的 np.long。
    """
    from scipy.spatial import cKDTree

    rotation, translation = transform[:3, :3], transform[:3, 3]
    src_in_ref = src_points @ rotation.T + translation
    tree = cKDTree(src_in_ref)
    indices_list = tree.query_ball_point(ref_points, matching_radius)
    corr_indices = [(i, j) for i, indices in enumerate(indices_list) for j in indices]
    if len(corr_indices) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(corr_indices, dtype=np.int64)


# 调试开关：为 True 时每次取样本都会弹出窗口显示两片点云（阻塞式，训练前请置 False）
SHOW_POINT_CLOUD = False


def show_point_clouds(ref_points, src_points, rotation, translation):
    """弹出 Open3D 窗口显示两片点云。

    ref 为红色；src 用真值变换对齐到 ref 坐标系后显示为蓝色。
    """
    import open3d as o3d

    src_in_ref = src_points @ rotation.T + translation
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
    ref_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_in_ref)
    src_pcd.paint_uniform_color([0.0, 0.6, 1.0])

    o3d.visualization.draw_geometries(
        [ref_pcd, src_pcd],
        window_name='Cross3DReg: ref=lidar(red)  src=array(blue, GT-aligned)',
    )


def estimate_overlap(ref_points, src_points, transform, radius):
    """估计重叠率：ref 点中在 radius 内能找到 src 对应点的比例。"""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return 1.0

    rotation, translation = transform[:3, :3], transform[:3, 3]
    src_in_ref = src_points @ rotation.T + translation
    distances, _ = cKDTree(src_in_ref).query(ref_points)
    return float(np.mean(distances < radius))


class Cross3DRegDataset(torch.utils.data.Dataset):
    r"""Cross3DReg 配准数据集。

    Args:
        dataset_root: 数据集根目录，默认 ``D:\ourdata\Cross3DReg``
        subset: 'train' / 'val' / 'test'（也接受 '3DMatch' / '3DLoMatch'，映射到 'test'）
        point_limit: 单帧点数上限，None 表示不限制
        use_augmentation: 是否做随机旋转 + 噪声增强
        augmentation_noise: 噪声幅度（米）
        augmentation_rotation: 随机旋转角度范围
        overlap_threshold: 只保留 overlap > 阈值的样本；metadata 无 overlap 字段时忽略
        return_corr_indices: 是否返回真值对应点索引
        matching_radius: 找对应点用的半径（return_corr_indices=True 时必填）
        rotated: 是否对两帧坐标分别施加随机旋转（训练扰动）
        invert_ground_truth: metadata['transform'] 是否为 ref->src 方向
            （Cross3DReg 实测为 array->lidar，即 src->ref，默认 False 无需求逆；
             3DMatch 风格的 rotation/translation 不受影响）
        compute_overlap: metadata 没有 overlap 字段时是否在线估计
        overlap_radius: 估计重叠率时的距离半径（米）
    """

    def __init__(
        self,
        dataset_root=None,
        subset='train',
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
        invert_ground_truth=False,
        compute_overlap=False,
        overlap_radius=0.05,
    ):
        super().__init__()


        dataset_root = DEFAULT_DATASET_ROOT
        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')

        self.point_limit = point_limit
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.overlap_threshold = overlap_threshold
        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')
        self.rotated = rotated
        self.invert_ground_truth = invert_ground_truth
        self.compute_overlap = compute_overlap
        self.overlap_radius = overlap_radius

        metadata_path = osp.join(self.metadata_root, f'test.pkl')
        if not osp.exists(metadata_path):
            raise FileNotFoundError(f'找不到 metadata 文件: {metadata_path}')
        self.metadata_list = load_metadata(metadata_path)

        if self.overlap_threshold is not None:
            if len(self.metadata_list) > 0 and 'overlap' in self.metadata_list[0]:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]
            else:
                warnings.warn('metadata 中没有 "overlap" 字段，overlap_threshold 被忽略。')

    def __len__(self):
        # return len(self.metadata_list)
        return 100



    def _load_point_cloud(self, relative_path):
        # metadata 里的路径使用 Windows 反斜杠，统一成 '/' 以保证跨平台可用
        file_path = osp.join(self.dataset_root, relative_path.replace('\\', '/'))
        points = read_ply_points(file_path)
        # points = voxel_downsample_to_target(file_path)
        return points

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""增强：ref_points = src_points @ rotation.T + translation。

        1. 随机旋转其中一个点云；
        2. 双帧加噪。
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def _get_rotation_translation(self, metadata):
        """返回 (R, t)，满足 ref_points = src_points @ R.T + t。"""
        if 'transform' in metadata:
            transform = np.asarray(metadata['transform'], dtype=np.float64)
            rotation, translation = transform[:3, :3], transform[:3, 3]
            if self.invert_ground_truth:
                rotation, translation = invert_transform(rotation, translation)
            return rotation, translation
        if 'rotation' in metadata and 'translation' in metadata:
            # 3DMatch 风格：已经是 src -> ref 方向，不需要求逆
            return (
                np.asarray(metadata['rotation'], dtype=np.float64),
                np.asarray(metadata['translation'], dtype=np.float64),
            )
        raise KeyError('metadata 中既没有 "transform" 也没有 "rotation"/"translation"')

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata.get('seq_id', metadata.get('scene_name', ''))
        data_dict['ref_frame'] = metadata.get('frame0', metadata.get('frag_id0'))
        data_dict['src_frame'] = metadata.get('frame1', metadata.get('frag_id1'))

        # get transformation (src -> ref)
        rotation, translation = self._get_rotation_translation(metadata)

        # get point cloud: ref = pcd1 (lidar, 全局基准), src = pcd0 (array, 待配准)
        ref_points = self._load_point_cloud(metadata['pcd1'])
        src_points = self._load_point_cloud(metadata['pcd0'])

        # 调试：弹出窗口查看两片点云（红=ref/lidar，蓝=src/array 按真值对齐）
        if SHOW_POINT_CLOUD:
            show_point_clouds(ref_points, src_points, rotation, translation)

        # augmentation
        # if self.use_augmentation:
        #     ref_points, src_points, rotation, translation = self._augment_point_cloud(
        #         ref_points, src_points, rotation, translation
        #     )

        # if self.rotated:
        #     ref_rotation = random_sample_rotation_v2()
        #     ref_points = np.matmul(ref_points, ref_rotation.T)
        #     rotation = np.matmul(ref_rotation, rotation)
        #     translation = np.matmul(ref_rotation, translation)
        #
        #     src_rotation = random_sample_rotation_v2()
        #     src_points = np.matmul(src_points, src_rotation.T)
        #     rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            corr_indices = find_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        # overlap: 优先用 metadata，其次在线估计，否则填 1.0
        if 'overlap' in metadata:
            overlap = float(metadata['overlap'])
        elif self.compute_overlap:
            overlap = estimate_overlap(ref_points, src_points, transform, self.overlap_radius)
        else:
            overlap = 1.0

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['overlap'] = overlap

        return data_dict


if __name__ == '__main__':
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_ROOT
    for split in ['train', 'val', 'test']:
        dataset = Cross3DRegDataset(root, split)
        sample = dataset[0]
        det = np.linalg.det(sample['transform'][:3, :3])
        print(
            f'[{split}] pairs={len(dataset):5d} '
            f'ref={sample["ref_points"].shape} src={sample["src_points"].shape} '
            f'scene={sample["scene_name"]} frames={sample["ref_frame"]}/{sample["src_frame"]} '
            f'det(R)={det:.4f} overlap={sample["overlap"]:.3f}'
        )
