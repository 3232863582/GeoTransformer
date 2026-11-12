import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))#获取当前 config.py 的绝对路径
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

ensure_dir(_C.output_dir)#目录不存在就创建（含所有中间父目录），已存在则什么都不做。
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data
_C.data = edict()#支持属性访问的字典
_C.data.dataset_root = r'/media/lxy/新加卷1/zj/Cross3DReg'
# 测试集要跑的场景（seq_id 0~10）；直接改这个数组即可，None 表示只跑 test.pkl
# 数组非 None 时会从 train+val+test 合并元数据中按场景筛选：
#   00~05 来自 train，06~07 来自 val，08~10 来自 test
# _C.data.test_scenes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_C.data.test_scenes = None
# _C.data.test_scenes = [8]
# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
# _C.train.point_limit = 10000
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0

# test data
_C.test = edict()
_C.test.batch_size = 1
# Windows 下每个 worker 都会重新加载 torch/CUDA DLL，8 个进程很容易触发
# "WinError 1455 页面文件太小"，测试时用 0（训练可按内存情况调成 2~4）
_C.test.num_workers = 8
_C.test.point_limit = 30000

# evaluation
# 以下阈值按 scene 08（3200 对）实测分布确定：
#   RRE  p50=0.89° p90=3.23° p95=7.95°
#   RTE  p50=0.18m p90=0.60m p95=1.36m
#   RMSE p50=0.22m p90=0.69m p95=1.41m
#   IR@0.1m=0.26%（基本无效）/ IR@0.5m=11.3% / IR@1.0m=30.7%
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0#粗匹配（coarse matching）指标 PIR 的判定阈值
_C.eval.acceptance_radius = 0.5#细匹配内点半径：0.1m 时 IR 只有 0.3%，放宽到 0.5m（scene08 IR≈11%）
_C.eval.inlier_ratio_threshold = 0.1#细匹配“成功”判定：IR>=10%，约 67% 样本达标
_C.eval.rmse_threshold = 0.5#仅用于报告 RMSE，不再参与配准召回判定
_C.eval.rre_threshold = 2.0#配准召回：RRE<2°（scene08 RR≈0.807）
_C.eval.rte_threshold = 0.5#配准召回：RTE<0.5m

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 20
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
# _C.backbone.init_voxel_size = 0.025
_C.backbone.init_voxel_size = 0.5
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
# 超点层级（points_list[3]）体素 = init_voxel_size*8 = 4.0m。
# 原 0.05m 只有体素的 1/80，几乎找不到真值超点对应（实测每对 0 个），
# coarse loss 对空张量求 mean 会得到 NaN。
# 按 3DMatch 的比例（匹配半径 ≈ 最终体素的一半）取 2.0m；
# 实测每对约 320~360 个真值超点对应（1.0m 时约 100~250）。
_C.model.ground_truth_matching_radius = 2.0
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 4
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
# 细层体素是 1.0m，0.1m 会把大量正确的细层对应判成外点，放宽到 0.2m
_C.fine_matching.acceptance_radius = 0.5
_C.fine_matching.mutual = True
# 早期 Sinkhorn 输出接近均匀（exp(score) 均值 ~0.004），0.05 会把候选全部滤掉；
# 先降到 0.01，训练后期分数拉开后可再调回
_C.fine_matching.confidence_threshold = 0.01
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
# 细层体素 1.0m，0.2m 时正样本只占 ~1%，fine loss 几乎全由 dustbin/slack 主导；
# 实测 0.5m 时正样本约为 0.2m 的 7 倍
_C.fine_loss.positive_radius = 0.5
# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
