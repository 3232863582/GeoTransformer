import argparse
import os.path as osp
import time
import warnings

import numpy as np

# PyTorch 2.x 在 logical_and 广播时会产生 "An output ... was resized" 的弃用警告，
# 消息里带形状导致无法按位置去重，每个样本都会重复刷屏。该警告不影响结果，直接屏蔽。
warnings.filterwarnings('ignore', message='An output with one or more elements was resized')

from geotransformer.engine import SingleTester
from geotransformer.utils.torch import release_cuda
from geotransformer.utils.common import ensure_dir, get_log_string

from dataset import test_data_loader
from config import make_cfg
from model import create_model
from loss import Evaluator


# ==================== 可视化开关（直接改这里）====================
VISUALIZE = False     # True: 每个样本弹窗显示输入点云和配准结果
VIS_SAMPLES = 10       # 最多可视化多少个样本
VIS_SCENE = None      # 只可视化指定场景（如 '08' / '10'）；None 表示不限制
# ================================================================


def _apply_transform_np(points, transform):
    """用 (4, 4) 变换矩阵变换 numpy 点云：p' = R p + t。"""
    rotation, translation = transform[:3, :3], transform[:3, 3]
    return np.asarray(points) @ rotation.T + translation


def _make_point_cloud(points, color):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    pcd.paint_uniform_color(color)
    return pcd


def visualize_input(scene_name, ref_id, src_id, ref_points, src_points):
    """配准前：ref 红色，src 蓝色（各自原始坐标系，未对齐）。"""
    import open3d as o3d

    ref_pcd = _make_point_cloud(ref_points, [1.0, 0.0, 0.0])
    src_pcd = _make_point_cloud(src_points, [0.0, 0.6, 1.0])
    o3d.visualization.draw_geometries(
        [ref_pcd, src_pcd],
        window_name=f'Input [{scene_name}] ref={ref_id}(red) src={src_id}(blue)',
    )


def visualize_result(scene_name, ref_points, src_points, estimated_transform, gt_transform):
    """配准后：ref 红色，估计变换后的 src 绿色，GT 变换后的 src 蓝色。"""
    import open3d as o3d

    ref_pcd = _make_point_cloud(ref_points, [1.0, 0.0, 0.0])
    est_pcd = _make_point_cloud(_apply_transform_np(src_points, estimated_transform), [0.0, 0.8, 0.2])
    gt_pcd = _make_point_cloud(_apply_transform_np(src_points, gt_transform), [0.0, 0.5, 1.0])
    o3d.visualization.draw_geometries(
        [ref_pcd, est_pcd, gt_pcd],
        window_name=f'Result [{scene_name}] red=ref green=estimated blue=GT',
    )


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'val'], help='test benchmark')
    return parser


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)
        self.vis_count = 0

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def visualize_sample(self, data_dict, output_dict):
        """按模块顶部 VISUALIZE / VIS_SAMPLES / VIS_SCENE 的配置弹窗。"""
        if not VISUALIZE:
            return
        scene_name = data_dict['scene_name']
        if VIS_SCENE is not None and scene_name != VIS_SCENE:
            return
        if self.vis_count >= VIS_SAMPLES:
            return

        ref_points = release_cuda(output_dict['ref_points'])
        src_points = release_cuda(output_dict['src_points'])
        estimated_transform = release_cuda(output_dict['estimated_transform'])
        gt_transform = release_cuda(data_dict['transform'])
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        visualize_input(scene_name, ref_id, src_id, ref_points, src_points)
        visualize_result(scene_name, ref_points, src_points, estimated_transform, gt_transform)

        self.vis_count += 1
        self.logger.info(f'Visualized {self.vis_count}/{VIS_SAMPLES}: {scene_name} {ref_id}_{src_id}')

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f'{ref_id}_{src_id}.npz')
        # 不压缩：单样本 savez_compressed 约 300ms，savez 约 10ms，
        # eval.py 读取也从 ~36ms 降到 ~6ms（多数字段本身已接近随机）。
        np.savez(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],
        )

        # 保存完成后再弹窗，关窗/中断不会导致该样本的 npz 丢失
        self.visualize_sample(data_dict, output_dict)


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
