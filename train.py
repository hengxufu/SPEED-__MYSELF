'''
MIT License

Copyright (c) 2021 SLAB Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.tensorboard import SummaryWriter

import os
import os.path as osp
import json
import logging
import math
from scipy.io import loadmat

from config import cfg
from src.styleaug.styleAugmentor import StyleAugmentor
from src.nets.build     import get_model, get_optimizer
from src.datasets.build import make_dataloader
from src.core.trainer   import train_single_epoch_krn, train_single_epoch_spn
from src.core.inference import valid_krn, valid_spn
from src.utils.utils    import setup_logger, set_all_seeds, \
                               save_checkpoint, load_checkpoint, \
                               load_tango_3d_keypoints, load_camera_intrinsics

logger = logging.getLogger(__name__)

_RESULT_HEADER_DONE = set()


def _count_trainable_named_params(model):
    stats = {
        'backbone_trainable_params': 0,
        'head_trainable_params': 0,
    }
    for name, p in model.named_parameters():
        if name.startswith('backbone.'):
            if p.requires_grad:
                stats['backbone_trainable_params'] += p.numel()
        else:
            if p.requires_grad:
                stats['head_trainable_params'] += p.numel()
    return stats


def _log_optimizer_param_groups(optimizer):
    groups = []
    for i, pg in enumerate(optimizer.param_groups):
        lr = float(pg.get('lr', float('nan')))
        n_params = 0
        for p in pg.get('params', []):
            try:
                n_params += int(p.numel())
            except Exception:
                pass
        groups.append({'index': i, 'lr': lr, 'numel': n_params})
    for g in groups:
        logger.info('[optimizer] group=%d lr=%.8g numel=%d', g['index'], g['lr'], g['numel'])
    return groups

def _append_eval_results(cfg, epoch, performances):
    if cfg.resultfn is None or cfg.resultfn == '':
        return
    if not osp.exists(cfg.logdir):
        os.makedirs(cfg.logdir)

    writefn = osp.join(cfg.logdir, cfg.resultfn)
    is_new = not osp.exists(writefn)
    metrics = list(performances.keys())
    with open(writefn, 'a', encoding='utf-8') as f:
        if writefn not in _RESULT_HEADER_DONE:
            has_header = False
            header_cols = None
            if not is_new:
                try:
                    with open(writefn, 'r', encoding='utf-8') as rf:
                        for _ in range(50):
                            line = rf.readline()
                            if not line:
                                break
                            if line.startswith('epoch\t') or line.strip() == 'epoch':
                                has_header = True
                                header_cols = line.strip().split('\t')
                                break
                except Exception:
                    has_header = False

            header = ['epoch']
            for metric in metrics:
                unit = performances[metric].unit if hasattr(performances[metric], 'unit') else '-'
                header.append(f'{metric} [{unit}]')

            if is_new or (not has_header):
                f.write('\t'.join(header) + '\n')
            else:
                # If the metrics set changed during experimentation, append a new header row for clarity.
                if header_cols is not None and header_cols != header:
                    f.write('\t'.join(header) + '\n')
            _RESULT_HEADER_DONE.add(writefn)

        row = [str(epoch)]
        for metric in metrics:
            val = performances[metric].avg
            row.append(f'{val:.6f}' if isinstance(val, (float, int)) else str(val))
        f.write('\t'.join(row) + '\n')

def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() and cfg.use_cuda else torch.device('cpu')

    # Logger
    setup_logger('train')

    # Seeds
    logger.info('Random seed value: {}'.format(cfg.seed))
    set_all_seeds(cfg.seed, cfg, True)

    # Save directory
    if not osp.exists(cfg.savedir): os.makedirs(cfg.savedir)
    logger.info('Checkpoints will be saved to {}'.format(cfg.savedir))

    # Tensorboard log directory
    if not osp.exists(cfg.logdir): os.makedirs(cfg.logdir)
    writer = SummaryWriter(cfg.logdir)
    logger.info('Tensorboard logs will be saved to {}'.format(cfg.logdir))

    # Save current config
    with open(osp.join(cfg.savedir, 'config.txt'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Pose estimation CNN
    model = get_model(cfg)
    logger.info('Model class: %s', model.__class__.__name__)
    logger.info('Backbone name: %s', getattr(model, 'backbone_name', getattr(cfg, 'backbone', '')))
    logger.info('Head type: %s', getattr(cfg, 'krn_head', ''))
    if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap':
        freeze_epochs = int(getattr(cfg, 'freeze_backbone_epochs', 0) or 0)
        bb_pre = bool(getattr(cfg, 'backbone_pretrained', False))
        if freeze_epochs > 0 and (not bb_pre):
            raise RuntimeError('freeze_backbone_epochs>0 requires backbone_pretrained=True.')
        if not str(getattr(cfg, 'backbone_pretrained_path', '') or '').strip():
            cfg.backbone_pretrained_path = osp.join(cfg.projroot, 'log', 'pretrained', f'{cfg.backbone}.pth')
        bb_pre_path = str(getattr(cfg, 'backbone_pretrained_path', '') or '')
        logger.info('[preflight] backbone_pretrained=%s', bb_pre)
        logger.info('[preflight] backbone_pretrained_path=%s', bb_pre_path)
        logger.info('[preflight] freeze_backbone_epochs=%d', freeze_epochs)
        if freeze_epochs > 0 and (not osp.exists(bb_pre_path)):
            raise RuntimeError(
                'freeze_backbone_epochs>0 but local backbone_pretrained_path was not found. '
                f'backbone_pretrained_path={bb_pre_path}. '
                'Provide --backbone_pretrained_path <file> or set --freeze_backbone_epochs 0.'
            )

    # Style Augmentor
    styleAugmentor = None
    if cfg.randomize_texture:
        styleAugmentor = StyleAugmentor(cfg.texture_alpha, device)
        logger.info('Texture randomization enabled with alpha = {}'.format(cfg.texture_alpha))
        logger.info('   - Randomization ratio: {:.2f}'.format(cfg.texture_ratio))

    # Optimizer
    optimizer = get_optimizer(cfg, model)
    for pg in optimizer.param_groups:
        if 'initial_lr' not in pg:
            pg['initial_lr'] = pg.get('lr', cfg.lr)

    if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap':
        _log_optimizer_param_groups(optimizer)

    # Load checkpoint
    checkpoint_file = osp.join(cfg.savedir, 'checkpoint.pth.tar')
    logger.info('Checkpoint path: %s', checkpoint_file)
    if cfg.auto_resume and osp.exists(checkpoint_file):
        last_epoch, best_speed = load_checkpoint(checkpoint_file, model, optimizer, device)
        begin_epoch = last_epoch
        best_perf   = begin_epoch
    else:
        begin_epoch = 0
        # best_perf   = 1e10
        best_perf   = begin_epoch

    # Model to device
    model = model.to(device)

    # Mixed-precision training?
    # - Using PyTorch AMP package, requires PyTorch 1.6 or above
    scaler = None
    if cfg.fp16:
        scaler = torch.cuda.amp.GradScaler()
        logger.info('Mixed-precision training enabled')

    lr_scheduler = None
    lr_schedule = getattr(cfg, 'lr_schedule', 'step')
    if lr_schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_alpha
        )
    else:
        def _set_lr(epoch_idx):
            warm = int(getattr(cfg, 'warmup_epochs', 0) or 0)
            max_epochs = int(cfg.max_epochs)
            min_lr = float(getattr(cfg, 'min_lr', 0.0) or 0.0)
            for pg in optimizer.param_groups:
                base_lr = float(pg.get('initial_lr', pg['lr']))
                if warm > 0 and epoch_idx < warm:
                    lr = base_lr * float(epoch_idx + 1) / float(warm)
                else:
                    t = 0.0
                    if max_epochs > warm:
                        t = float(epoch_idx - warm) / float(max_epochs - warm)
                        t = min(max(t, 0.0), 1.0)
                    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
                pg['lr'] = lr

    # Dataloader
    train_loader = make_dataloader(cfg, is_train=True,  is_source=True)
    test_loader  = make_dataloader(cfg, is_train=False, is_source=False)

    # Miscellaneous items
    corners3D = load_tango_3d_keypoints(cfg.keypts_3d_model)
    cameraMatrix, distCoeffs = load_camera_intrinsics(
                osp.join(cfg.dataroot, cfg.dataname, 'camera.json'))
    attClasses = loadmat(cfg.attitude_class)['qClass'] # [Nclasses x 4]
    assert attClasses.shape[0] == cfg.num_classes, 'Number of classes not matching.'

    # Main loop
    perf    = 1e10
    is_best = False
    for epoch in range(begin_epoch, cfg.max_epochs):
        if lr_schedule != 'step':
            _set_lr(epoch)
        if hasattr(model, 'set_backbone_trainable'):
            freeze_epochs = int(getattr(cfg, 'freeze_backbone_epochs', 0) or 0)
            model.set_backbone_trainable(epoch >= freeze_epochs)
            if freeze_epochs > 0 and (epoch + 1) <= freeze_epochs:
                stats = _count_trainable_named_params(model)
                if int(stats['backbone_trainable_params']) != 0:
                    raise RuntimeError(
                        f'Expected epoch {epoch+1}-{freeze_epochs} backbone trainable params = 0, '
                        f'but got {stats["backbone_trainable_params"]}.'
                    )

        # Train an epoch
        eval('train_single_epoch_'+cfg.model_name)(
                epoch+1, cfg, model, train_loader, optimizer, writer,
                device, styleAugmentor=styleAugmentor, scaler=scaler)

        # Update LR scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Test
        if (epoch+1) % cfg.test_epoch == 0 and cfg.test_epoch > 0:
            performances = eval('valid_'+cfg.model_name)(
                epoch+1, cfg, model, test_loader,
                cameraMatrix, distCoeffs, corners3D, writer, device, attClasses)
            _append_eval_results(cfg, epoch+1, performances)

        # Save best models every epoch
        perf = epoch+1
        if perf > best_perf:
            best_perf = perf
            is_best = True
        else:
            is_best = False

        # Save
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.model_name,
            'state_dict': model.state_dict(),
            'best_score': best_perf,
            'optimizer': optimizer.state_dict(),
        }, is_best, cfg.savedir)

    # Close tensorboard
    writer.close()

if __name__=='__main__':
    main()
