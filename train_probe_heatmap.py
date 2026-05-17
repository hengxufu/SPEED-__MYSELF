from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import logging

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from config import cfg
from src.nets.build import get_model, get_optimizer
from src.datasets.build import build_dataset
from src.datasets.transforms import build_transforms
from src.core.trainer import train_single_epoch_krn
from src.core.inference import valid_krn
from src.utils.utils import setup_logger, set_all_seeds, load_tango_3d_keypoints, load_camera_intrinsics, save_checkpoint
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def _count_trainable_named_params(model):
    stats = {
        'backbone_trainable_params': 0,
        'head_trainable_params': 0,
        'backbone_total_params': 0,
        'head_total_params': 0,
    }
    for name, p in model.named_parameters():
        is_backbone = name.startswith('backbone.')
        if is_backbone:
            stats['backbone_total_params'] += p.numel()
            if p.requires_grad:
                stats['backbone_trainable_params'] += p.numel()
        else:
            stats['head_total_params'] += p.numel()
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


def _resolve_root_and_csv(dataroot, dataname, domain, csv_rel):
    dr = osp.abspath(dataroot)
    if osp.isabs(csv_rel):
        if osp.exists(csv_rel):
            base = osp.join(dr, dataname) if dataname else dr
            base = base if osp.exists(base) else dr
            return base, csv_rel
        raise FileNotFoundError('CSV not found: {}'.format(csv_rel))

    candidates = []
    if dataname:
        candidates.append(osp.join(dr, dataname))
    candidates.append(dr)

    tried = []
    for root in candidates:
        csv_path = osp.join(root, domain, 'splits_{}'.format(cfg.model_name), csv_rel)
        tried.append(csv_path)
        if osp.exists(csv_path):
            return root, csv_path
    raise FileNotFoundError('CSV not found. Tried:\n' + '\n'.join(tried))


class EvalCSV(Dataset):
    def __init__(self, root, csv_path, num_keypoints=11, transforms=None, limit=0):
        import pandas as pd
        self.root = root
        self.num_keypoints = int(num_keypoints)
        self.transforms = transforms
        df = pd.read_csv(csv_path, header=None)
        if int(limit) > 0:
            df = df.iloc[:int(limit)]
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        row = self.df.iloc[idx].to_numpy()
        imgpath = osp.join(self.root, row[0])
        img = Image.open(imgpath).convert('RGB')
        bbox = np.array(row[1:5], dtype=np.float32)
        q_gt = np.array(row[5:9], dtype=np.float32)
        t_gt = np.array(row[9:12], dtype=np.float32)
        keypts = np.array(row[12:], dtype=np.float32).reshape(self.num_keypoints, 2).T  # (2,K) pix

        if self.transforms is not None:
            img, bbox, keypts = self.transforms(img, bbox, keypts)

        return img, torch.as_tensor(bbox, dtype=torch.float32), torch.as_tensor(keypts, dtype=torch.float32), torch.as_tensor(q_gt), torch.as_tensor(t_gt)


def _append_probe_results(out_path, epoch, train_metrics, val_metrics, train_loss):
    if not osp.exists(osp.dirname(out_path)):
        os.makedirs(osp.dirname(out_path))
    keys = []
    for k in sorted(train_metrics.keys()):
        keys.append('train_' + k)
    for k in sorted(val_metrics.keys()):
        keys.append('val_' + k)
    keys.extend(['train_loss_hm', 'train_pos_loss', 'train_neg_loss'])

    row_map = {}
    for k in sorted(train_metrics.keys()):
        row_map['train_' + k] = float(train_metrics[k].avg)
    for k in sorted(val_metrics.keys()):
        row_map['val_' + k] = float(val_metrics[k].avg)
    row_map['train_loss_hm'] = float(train_loss.get('loss_hm', float('nan')))
    row_map['train_pos_loss'] = float(train_loss.get('pos_loss', float('nan')))
    row_map['train_neg_loss'] = float(train_loss.get('neg_loss', float('nan')))

    if not osp.exists(out_path):
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('epoch\t' + '\t'.join(keys) + '\n')
            row = [str(int(epoch))] + ['{:.6f}'.format(float(row_map.get(k, float('nan')))) for k in keys]
            f.write('\t'.join(row) + '\n')
        return

    with open(out_path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
        old_cols = header.split('\t')
        old_keys = old_cols[1:]
        lines = [ln.rstrip('\n') for ln in f if ln.strip() != '']

    if list(old_keys) == list(keys):
        with open(out_path, 'a', encoding='utf-8') as f:
            row = [str(int(epoch))] + ['{:.6f}'.format(float(row_map.get(k, float('nan')))) for k in keys]
            f.write('\t'.join(row) + '\n')
        return

    old_key_set = set(old_keys)
    new_key_set = set(keys)
    merged_keys = list(keys) + sorted([k for k in old_keys if k not in new_key_set])

    by_epoch = {}
    for ln in lines:
        parts = ln.split('\t')
        if len(parts) < 1:
            continue
        try:
            ep = int(float(parts[0]))
        except Exception:
            continue
        vals = parts[1:]
        d = {}
        for k, v in zip(old_keys, vals):
            try:
                d[k] = float(v)
            except Exception:
                d[k] = float('nan')
        by_epoch[ep] = d

    by_epoch[int(epoch)] = row_map
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('epoch\t' + '\t'.join(merged_keys) + '\n')
        for ep in sorted(by_epoch.keys()):
            d = by_epoch[ep]
            row = [str(int(ep))] + ['{:.6f}'.format(float(d.get(k, float('nan')))) for k in merged_keys]
            f.write('\t'.join(row) + '\n')


def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() and cfg.use_cuda else torch.device('cpu')
    setup_logger('train_probe_heatmap')
    logger.info('Random seed value: {}'.format(cfg.seed))
    set_all_seeds(cfg.seed, cfg, True)

    if not osp.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    with open(osp.join(cfg.logdir, 'config.txt'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap':
        expected_backbone = 'swin_tiny_patch4_window7_224'
        if str(getattr(cfg, 'backbone', '')) != expected_backbone:
            raise RuntimeError(f'Probe expects backbone={expected_backbone}, got backbone={getattr(cfg, "backbone", "")}')

        cfg.p_aug = 0.0
        cfg.deterministic_crop = True
        cfg.train_subset_size = int(getattr(cfg, 'train_subset_size', 512) or 512)
        cfg.val_subset_size = int(getattr(cfg, 'val_subset_size', 500) or 500)

        freeze_epochs = int(getattr(cfg, 'freeze_backbone_epochs', 0) or 0)
        if freeze_epochs != 5:
            raise RuntimeError(f'Probe expects freeze_backbone_epochs=5, got {freeze_epochs}')

        if bool(getattr(cfg, 'backbone_pretrained', False)) is not True:
            raise RuntimeError('Probe expects backbone_pretrained=True (ImageNet pretrained Swin Tiny).')

        if not str(getattr(cfg, 'backbone_pretrained_path', '') or '').strip():
            cfg.backbone_pretrained_path = osp.join(cfg.projroot, 'log', 'pretrained', f'{cfg.backbone}.pth')
        bb_path = str(getattr(cfg, 'backbone_pretrained_path', '') or '')

        logger.info('[preflight] backbone=%s', getattr(cfg, 'backbone', ''))
        logger.info('[preflight] backbone_pretrained=%s', bool(getattr(cfg, 'backbone_pretrained', False)))
        logger.info('[preflight] backbone_pretrained_path=%s', bb_path)
        logger.info('[preflight] freeze_backbone_epochs=%d', freeze_epochs)
        logger.info('[preflight] train_subset_size=%d val_subset_size=%d', int(cfg.train_subset_size), int(cfg.val_subset_size))
        logger.info('[preflight] p_aug=%.3f deterministic_crop=%s', float(getattr(cfg, 'p_aug', 0.0)), bool(getattr(cfg, 'deterministic_crop', False)))

        if freeze_epochs > 0 and (not osp.exists(bb_path)):
            raise RuntimeError(
                'freeze_backbone_epochs>0 but local Swin pretrained weights were not found. '
                f'Expected backbone_pretrained_path={bb_path}. '
                'Provide --backbone_pretrained_path <file> or set --freeze_backbone_epochs 0 (random backbone debug run only).'
            )

    model = get_model(cfg).to(device)
    if cfg.pretrained and cfg.pretrained != '':
        state = torch.load(cfg.pretrained, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        if isinstance(state, dict):
            model.load_state_dict(state, strict=True)
            logger.info('Loaded pretrained weights from %s', cfg.pretrained)

    if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap':
        freeze_epochs = int(getattr(cfg, 'freeze_backbone_epochs', 0) or 0)
        bb_pre = bool(getattr(cfg, 'backbone_pretrained', False))
        bb_pre_path = str(getattr(cfg, 'backbone_pretrained_path', '') or '')
        if freeze_epochs > 0 and (not bb_pre):
            raise RuntimeError('freeze_backbone_epochs>0 requires backbone_pretrained=True.')
        if freeze_epochs > 0 and (not osp.exists(bb_pre_path)):
            raise RuntimeError(
                'freeze_backbone_epochs>0 but local backbone_pretrained_path was not found. '
                f'backbone_pretrained_path={bb_pre_path}. '
                'Provide --backbone_pretrained_path <file> or set --freeze_backbone_epochs 0.'
            )

        bb_info = getattr(getattr(model, 'backbone', None), 'pretrained_load_info', None)
        if bb_info is None:
            raise RuntimeError('Backbone did not report pretrained_load_info; cannot verify ImageNet weights were loaded.')
        logger.info(
            '[pretrain] matched=%d/%d missing=%d unexpected=%d',
            int(bb_info.get('matched_num_keys', -1)),
            int(bb_info.get('model_num_keys', -1)),
            int(bb_info.get('missing_num_keys', -1)),
            int(bb_info.get('unexpected_num_keys', -1)),
        )
        matched = float(bb_info.get('matched_num_keys', 0) or 0)
        model_keys = float(bb_info.get('model_num_keys', 1) or 1)
        unexpected = float(bb_info.get('unexpected_num_keys', 0) or 0)
        if (matched / max(model_keys, 1.0)) < 0.1:
            raise RuntimeError('Backbone pretrained checkpoint key match ratio < 10%; likely wrong checkpoint.')
        if (unexpected / max(model_keys, 1.0)) > 0.5:
            raise RuntimeError('Backbone pretrained checkpoint has too many unexpected keys; likely incompatible checkpoint.')

    optimizer = get_optimizer(cfg, model)
    for pg in optimizer.param_groups:
        if 'initial_lr' not in pg:
            pg['initial_lr'] = pg.get('lr', cfg.lr)

    if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap':
        groups = _log_optimizer_param_groups(optimizer)
        if len(groups) < 2:
            raise RuntimeError('Heatmap KRN optimizer must have separate backbone/head param_groups.')
        lr0 = float(groups[0]['lr'])
        lr1 = float(groups[1]['lr'])
        if abs(lr0 - float(getattr(cfg, 'lr_backbone', 1e-5))) > 1e-12:
            raise RuntimeError(f'Optimizer backbone lr mismatch: group0 lr={lr0} cfg.lr_backbone={getattr(cfg, "lr_backbone", None)}')
        if abs(lr1 - float(getattr(cfg, 'lr_head', 1e-4))) > 1e-12:
            raise RuntimeError(f'Optimizer head lr mismatch: group1 lr={lr1} cfg.lr_head={getattr(cfg, "lr_head", None)}')

        if hasattr(model, 'set_backbone_trainable'):
            model.set_backbone_trainable(False)
        stats = _count_trainable_named_params(model)
        logger.info(
            '[params] backbone_trainable=%d head_trainable=%d backbone_total=%d head_total=%d',
            int(stats['backbone_trainable_params']),
            int(stats['head_trainable_params']),
            int(stats['backbone_total_params']),
            int(stats['head_total_params']),
        )
        if int(getattr(cfg, 'freeze_backbone_epochs', 0) or 0) > 0 and int(stats['backbone_trainable_params']) != 0:
            raise RuntimeError('Expected backbone trainable params = 0 during freeze_backbone_epochs, but found non-zero.')
        if int(stats['head_trainable_params']) <= 0:
            raise RuntimeError('Expected head trainable params > 0, but found 0.')

    train_ds = build_dataset(cfg, is_train=True, is_source=True, load_labels=True)
    train_n = int(getattr(cfg, 'train_subset_size', 512))
    train_n = min(train_n, len(train_ds))
    train_subset = Subset(train_ds, list(range(train_n)))
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Eval datasets (train/val subsets)
    eval_tf = build_transforms(
        cfg.model_name,
        cfg.input_shape,
        p_aug=float(getattr(cfg, 'p_aug', 0.0)),
        is_train=False,
        deterministic_crop=bool(getattr(cfg, 'deterministic_crop', True)),
    )

    root, train_csv = _resolve_root_and_csv(cfg.dataroot, cfg.dataname, cfg.train_domain, cfg.train_csv)
    root2, val_csv = _resolve_root_and_csv(cfg.dataroot, cfg.dataname, cfg.test_domain, cfg.test_csv)
    train_eval_ds = EvalCSV(root, train_csv, num_keypoints=cfg.num_keypoints, transforms=eval_tf, limit=train_n)
    val_n = int(getattr(cfg, 'val_subset_size', 500))
    val_eval_ds = EvalCSV(root2, val_csv, num_keypoints=cfg.num_keypoints, transforms=eval_tf, limit=val_n)

    train_eval_loader = DataLoader(train_eval_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    val_eval_loader = DataLoader(val_eval_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    corners3D = load_tango_3d_keypoints(cfg.keypts_3d_model)
    cameraMatrix, distCoeffs = load_camera_intrinsics(
        osp.join(cfg.dataroot, cfg.dataname, 'camera.json'))
    attClasses = loadmat(cfg.attitude_class)['qClass']

    result_file = osp.join(cfg.logdir, 'probe_results.txt')
    ckpt_every = int(getattr(cfg, 'ckpt_every', 5) or 5)
    ckpt_dir = osp.join(cfg.logdir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(cfg.max_epochs):
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

        train_loss = train_single_epoch_krn(
            epoch + 1, cfg, model, train_loader, optimizer, writer=None, device=device
        )

        train_perf = valid_krn(
            epoch + 1, cfg, model, train_eval_loader,
            cameraMatrix, distCoeffs, corners3D, writer=None, device=device, qClass=attClasses
        )
        val_perf = valid_krn(
            epoch + 1, cfg, model, val_eval_loader,
            cameraMatrix, distCoeffs, corners3D, writer=None, device=device, qClass=attClasses
        )

        _append_probe_results(result_file, epoch + 1, train_perf, val_perf, train_loss)

        logger.info('[probe] epoch=%d train_rmse_med=%.3f val_rmse_med=%.3f train_loss_hm=%.6f pos=%.6f neg=%.6f',
                    epoch + 1,
                    train_perf['keypoint_rmse_px_median'].avg if 'keypoint_rmse_px_median' in train_perf else float('nan'),
                    val_perf['keypoint_rmse_px_median'].avg if 'keypoint_rmse_px_median' in val_perf else float('nan'),
                    float(train_loss.get('loss_hm', float('nan'))),
                    float(train_loss.get('pos_loss', float('nan'))),
                    float(train_loss.get('neg_loss', float('nan'))))

        if ckpt_every > 0 and ((epoch + 1) % ckpt_every == 0):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': cfg.model_name,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'cfg': cfg.__dict__,
                },
                is_best=False,
                output_dir=ckpt_dir,
                filename=f'checkpoint_epoch_{epoch+1:03d}.pth.tar',
            )


if __name__ == '__main__':
    main()
