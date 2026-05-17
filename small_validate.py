from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import logging
import json

import torch

from config import cfg
from src.nets.build import get_model
from src.datasets.build import make_dataloader
from src.core.inference import valid_krn, valid_spn
from src.utils.utils import setup_logger, set_all_seeds, load_tango_3d_keypoints, load_camera_intrinsics
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def _load_weights(model, path, device):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    if isinstance(state, dict):
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError('Unsupported checkpoint format')
    model.to(device)
    logger.info('Loaded weights from {}'.format(path))


def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() and cfg.use_cuda else torch.device('cpu')
    setup_logger('small_validate')
    logger.info('Random seed value: {}'.format(cfg.seed))
    set_all_seeds(cfg.seed, cfg, True)

    if not osp.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    with open(osp.join(cfg.logdir, 'config.txt'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    model = get_model(cfg).to(device)
    if cfg.pretrained and cfg.pretrained != '':
        _load_weights(model, cfg.pretrained, device)

    test_loader = make_dataloader(cfg, is_train=False, is_source=False)

    corners3D = load_tango_3d_keypoints(cfg.keypts_3d_model)
    cameraMatrix, distCoeffs = load_camera_intrinsics(
        osp.join(cfg.dataroot, cfg.dataname, 'camera.json'))
    attClasses = loadmat(cfg.attitude_class)['qClass']

    epoch = int(getattr(cfg, 'eval_epoch', 0) or 0)
    performances = eval('valid_' + cfg.model_name)(
        epoch, cfg, model, test_loader, cameraMatrix, distCoeffs, corners3D, writer=None, device=device, qClass=attClasses
    )
    keys = list(performances.keys())
    out = {}
    for k in keys:
        v = performances[k]
        out[k] = float(v.avg) if hasattr(v, 'avg') else float(getattr(v, 'avg', v))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
