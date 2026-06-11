import argparse

import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))

def _default_os_key():
    if os.name == 'nt':
        return 'windows'
    if sys.platform == 'darwin':
        return 'mac'
    return 'linux'

_os_key = _default_os_key()

PROJROOTDIR = {'mac':  '/Users/taehapark/SLAB/speedplusbaseline',
               'linux': '/media/shared/Jeff/SLAB/speedplusbaseline',
               'windows':'D:/进阶项目/CNN/speedplusbaseline-master'}

DATAROOTDIR = {'mac':  '/Users/taehapark/SLAB/speedplus/data/datasets',
               'linux': '/home/jeffpark/SLAB/Dataset',
               'windows':'D:/进阶项目/CNN/speedplusv2'}

# Keep Windows defaults portable even when the repository is moved.
PROJROOTDIR['windows'] = _HERE
DATAROOTDIR['windows'] = osp.abspath(osp.join(_HERE, '..', 'speedplusv2'))

parser = argparse.ArgumentParser('Configurations for SPEED+ Baseline Study')

# ------------------------------------------------------------------------------------------
# Basic directories and names
parser.add_argument('--seed',     type=int, default=2021)
parser.add_argument('--projroot', type=str, default=PROJROOTDIR.get(_os_key, PROJROOTDIR['linux']))
parser.add_argument('--dataroot', type=str, default=DATAROOTDIR.get(_os_key, DATAROOTDIR['linux']))
parser.add_argument('--dataname', type=str, default='speedplus')
parser.add_argument('--savedir',  type=str, default='checkpoints/synthetic/krn')
parser.add_argument('--resultfn', type=str, default='')
parser.add_argument('--logdir',   type=str, default='log/synthetic/krn')
parser.add_argument('--pretrained', type=str, default='')

# ------------------------------------------------------------------------------------------
# Model config.
parser.add_argument('--model_name',      type=str,   default='krn')
parser.add_argument('--input_shape',     nargs='+',  type=int, default=(224, 224))
parser.add_argument('--num_keypoints',   type=int,   default=11)   # KRN-specific
parser.add_argument('--num_classes',     type=int,   default=5000) # SPN-specific
parser.add_argument('--num_neighbors',   type=int,   default=5)    # SPN-specific
parser.add_argument('--keypts_3d_model', type=str,   default='src/utils/tangoPoints.mat')
parser.add_argument('--attitude_class',  type=str,   default='src/utils/attitudeClasses.mat')
parser.add_argument('--backbone',        type=str,   default='')
parser.add_argument('--no_backbone_pretrained', dest='backbone_pretrained', action='store_false', default=True)
parser.add_argument('--backbone_pretrained_path', type=str, default='')
parser.add_argument('--debug_shapes',    dest='debug_shapes', action='store_true', default=False)
parser.add_argument('--krn_head',        type=str, default='direct', choices=['direct', 'heatmap'])
parser.add_argument('--heatmap_size',    nargs='+', type=int, default=(56, 56))
parser.add_argument('--heatmap_sigma',   type=float, default=2.0)
parser.add_argument('--heatmap_decode',  type=str, default='argmax', choices=['argmax', 'softargmax'])
parser.add_argument('--heatmap_beta',    type=float, default=100.0)
parser.add_argument('--heatmap_activation', type=str, default='none', choices=['sigmoid', 'none'])
parser.add_argument('--heatmap_loss', type=str, default='bce', choices=['bce', 'mse', 'heatmap_ce_coord_aux'])
parser.add_argument('--heatmap_pos_thr', type=float, default=0.1)
parser.add_argument('--heatmap_neg_weight', type=float, default=0.01)
parser.add_argument('--heatmap_pos_weight', type=float, default=0.0)
parser.add_argument('--heatmap_bg_weight', type=float, default=1.0)
parser.add_argument('--coord_aux_weight', type=float, default=0.1)
parser.add_argument('--heatmap_hard_kpts', type=str, default='3,4,5,6,10')
parser.add_argument('--heatmap_hard_kpt_weight', type=float, default=1.5)
parser.add_argument('--heatmap_other_kpt_weight', type=float, default=1.0)
parser.add_argument('--backbone_out_indices', type=str, default='')
parser.add_argument('--backbone_fpn', dest='backbone_fpn', action='store_true', default=False)
parser.add_argument('--z_min', type=float, default=0.0)
parser.add_argument('--ransac_reproj_thr_px', type=float, default=8.0)
parser.add_argument('--pose_reproj_thr_px', type=float, default=10.0)
parser.add_argument('--pose_t_min', type=float, default=0.5)
parser.add_argument('--pose_t_max', type=float, default=20.0)
parser.add_argument('--p_aug', type=float, default=-1.0)
parser.add_argument('--backbone_out_index', type=int, default=-1)
parser.add_argument('--input_normalize', type=str, default='')
parser.add_argument('--freeze_backbone_epochs', type=int, default=5)
parser.add_argument('--lr_backbone',     type=float, default=1e-5)
parser.add_argument('--lr_head',         type=float, default=1e-4)
parser.add_argument('--warmup_epochs',   type=int, default=5)
parser.add_argument('--min_lr',          type=float, default=0.0)
parser.add_argument('--grad_clip_norm',  type=float, default=1.0)
parser.add_argument('--debug_save_n',    type=int, default=20)
parser.add_argument('--return_test_keypts', dest='return_test_keypts', action='store_true', default=False)
parser.add_argument('--eval_epoch', type=int, default=0)
parser.add_argument('--train_subset_size', type=int, default=512)
parser.add_argument('--val_subset_size', type=int, default=500)

# ------------------------------------------------------------------------------------------
# Training config.
parser.add_argument('--start_over',        dest='auto_resume', action='store_false', default=True)
parser.add_argument('--randomize_texture', dest='randomize_texture', action='store_true', default=False)
parser.add_argument('--perform_dann',      dest='dann', action='store_true', default=False)
parser.add_argument('--texture_alpha',   type=float, default=0.5)
parser.add_argument('--texture_ratio',   type=float, default=0.5)
parser.add_argument('--use_fp16',          dest='fp16', action='store_true', default=False)
parser.add_argument('--batch_size',      type=int,   default=32)
parser.add_argument('--max_epochs',      type=int,   default=75)
parser.add_argument('--num_workers',     type=int,   default=8)
parser.add_argument('--test_epoch',      type=int,   default=-1)
parser.add_argument('--ckpt_every',     type=int,   default=5)
parser.add_argument('--optimizer',       type=str,   default='rmsprop')
parser.add_argument('--lr',              type=float, default=0.001)
parser.add_argument('--momentum',        type=float, default=0.9)
parser.add_argument('--weight_decay',    type=float, default=5e-5)
parser.add_argument('--lr_decay_alpha',  type=float, default=0.96)
parser.add_argument('--lr_decay_step',   type=int,   default=1)
parser.add_argument('--lr_schedule',     type=str,   default='step', choices=['step', 'cosine'])

# ------------------------------------------------------------------------------------------
# Dataset-related inputs
parser.add_argument('--train_domain', type=str, default='synthetic')
parser.add_argument('--test_domain',  type=str, default='lightbox')
parser.add_argument('--train_csv',    type=str, default='train.csv')
parser.add_argument('--test_csv',     type=str, default='lightbox.csv')
parser.add_argument('--max_eval_samples', type=int, default=0)
parser.add_argument('--deterministic_crop', dest='deterministic_crop', action='store_true', default=False)

# ------------------------------------------------------------------------------------------
# Other miscellaneous settings
parser.add_argument('--gpu_id',  type=int, default=0)
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', default=True)

# End
cfg = parser.parse_args()

cfg.projroot = osp.abspath(cfg.projroot)
cfg.dataroot = osp.abspath(cfg.dataroot)
if cfg.backbone is None or cfg.backbone == '':
    cfg.backbone = 'mobilenet_v2' if cfg.model_name == 'krn' else 'alexnet_bvlc'
if cfg.input_normalize == '':
    cfg.input_normalize = 'none' if cfg.backbone in ('mobilenet_v2', 'alexnet_bvlc', 'bvlc_alexnet') else 'imagenet'
if (cfg.resultfn is None or cfg.resultfn == '') and int(getattr(cfg, 'test_epoch', -1) or -1) > 0:
    cfg.resultfn = 'results.txt'
if cfg.model_name == 'krn' and cfg.krn_head == 'heatmap':
    cfg.return_test_keypts = True
if float(getattr(cfg, 'p_aug', -1.0)) < 0:
    if cfg.model_name == 'krn' and cfg.krn_head == 'heatmap':
        cfg.p_aug = 0.0
    else:
        cfg.p_aug = 0.5
if int(getattr(cfg, 'backbone_out_index', -1)) < 0:
    if cfg.model_name == 'krn' and cfg.krn_head == 'heatmap' and isinstance(cfg.backbone, str) and ('swin' in cfg.backbone):
        cfg.backbone_out_index = 2
    else:
        cfg.backbone_out_index = 3

if isinstance(getattr(cfg, 'backbone_out_indices', ''), str) and cfg.backbone_out_indices.strip() != '':
    try:
        cfg.backbone_out_indices = [int(x) for x in cfg.backbone_out_indices.replace(' ', '').split(',') if x != '']
    except Exception:
        cfg.backbone_out_indices = []
else:
    cfg.backbone_out_indices = []

if bool(getattr(cfg, 'backbone_fpn', False)) and len(getattr(cfg, 'backbone_out_indices', [])) == 0:
    if cfg.model_name == 'krn' and cfg.krn_head == 'heatmap' and isinstance(cfg.backbone, str) and ('swin' in cfg.backbone):
        cfg.backbone_out_indices = [1, 2, 3]
if cfg.dataname and not osp.exists(osp.join(cfg.dataroot, cfg.dataname)):
    has_camera_at_root = osp.exists(osp.join(cfg.dataroot, 'camera.json'))
    has_domain_at_root = any(osp.isdir(osp.join(cfg.dataroot, d)) for d in ('synthetic', 'lightbox', 'sunlamp'))
    if has_camera_at_root and has_domain_at_root:
        cfg.dataname = ''

def _maybe_abspath(base_dir, path):
    if path is None or path == '':
        return path
    if isinstance(path, str) and (path.startswith('http://') or path.startswith('https://')):
        return path
    return path if osp.isabs(path) else osp.join(base_dir, path)

cfg.keypts_3d_model = _maybe_abspath(cfg.projroot, cfg.keypts_3d_model)
cfg.attitude_class = _maybe_abspath(cfg.projroot, cfg.attitude_class)
cfg.savedir = _maybe_abspath(cfg.projroot, cfg.savedir)
cfg.logdir = _maybe_abspath(cfg.projroot, cfg.logdir)
cfg.pretrained = _maybe_abspath(cfg.projroot, cfg.pretrained)
cfg.backbone_pretrained_path = _maybe_abspath(cfg.projroot, cfg.backbone_pretrained_path)

# Convenience: allow placing backbone weights under log/pretrained without extra flags.
# If the file exists, prefer it to avoid network downloads for timm pretrained weights.
if (
    cfg.model_name == 'krn'
    and getattr(cfg, 'krn_head', 'direct') == 'heatmap'
    and isinstance(cfg.backbone, str)
    and (cfg.backbone_pretrained_path is None or cfg.backbone_pretrained_path == '')
):
    _cand = osp.join(cfg.projroot, 'log', 'pretrained', f'{cfg.backbone}.pth')
    if osp.exists(_cand) and osp.getsize(_cand) > 0:
        cfg.backbone_pretrained_path = _cand

# If a full model checkpoint is provided, backbone ImageNet weights are unnecessary.
# Avoid network downloads during evaluation unless an explicit backbone_pretrained_path is given.
if cfg.pretrained and cfg.pretrained != '' and (cfg.backbone_pretrained_path is None or cfg.backbone_pretrained_path == ''):
    cfg.backbone_pretrained = False
