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

import logging
import time
import random
import numpy as np

from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.cuda.amp import autocast

from src.nets.spn        import softmax_cross_entropy_with_logits
from src.utils.utils     import AverageMeter, report_progress
from src.utils.visualize import imshow, plot_2D_bbox, scatter_keypoints

logger = logging.getLogger("Training")
_TRAIN_DEBUG_ONCE = False
_TRAIN_DEBUG_BATCH_ONCE = False

def _count_trainable_params(module):
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable

def _debug_print_startup(cfg, model, optimizer):
    global _TRAIN_DEBUG_ONCE
    if _TRAIN_DEBUG_ONCE:
        return
    _TRAIN_DEBUG_ONCE = True

    model_class = model.__class__.__name__
    backbone_name = getattr(model, 'backbone_name', getattr(cfg, 'backbone', ''))
    head_type = getattr(cfg, 'krn_head', getattr(model, 'krn_head', ''))
    freeze_epochs = int(getattr(cfg, 'freeze_backbone_epochs', 0) or 0)
    det_crop = bool(getattr(cfg, 'deterministic_crop', False))
    p_aug = float(getattr(cfg, 'p_aug', 0.0))
    bb_pre = bool(getattr(cfg, 'backbone_pretrained', False))
    bb_pre_path = getattr(cfg, 'backbone_pretrained_path', '')
    init_ckpt = getattr(cfg, 'pretrained', '')

    bb = getattr(model, 'backbone', None)
    head = None
    if hasattr(model, 'head'):
        head = model.head

    bb_total, bb_train = _count_trainable_params(bb) if bb is not None else (0, 0)
    head_total, head_train = _count_trainable_params(head) if head is not None else (0, 0)

    pg = []
    for i, g in enumerate(optimizer.param_groups):
        lr = g.get('lr', None)
        wd = g.get('weight_decay', None)
        pg.append({'idx': i, 'lr': float(lr) if lr is not None else None, 'wd': float(wd) if wd is not None else None, 'n_params': len(g.get('params', []))})

    logger.info('[train_debug] model_class=%s backbone=%s head=%s', model_class, backbone_name, head_type)
    logger.info('[train_debug] optimizer_groups=%s', pg)
    logger.info('[train_debug] backbone_params=%d trainable=%d head_params=%d trainable=%d',
                bb_total, bb_train, head_total, head_train)
    logger.info('[train_debug] cfg: freeze_backbone_epochs=%d deterministic_crop=%s p_aug=%.3f', freeze_epochs, det_crop, p_aug)
    logger.info('[train_debug] cfg: backbone_pretrained=%s backbone_pretrained_path=%s init_pretrained_ckpt=%s',
                bb_pre, str(bb_pre_path), str(init_ckpt))
    if bb is not None:
        bb_impl = bb.__class__.__name__
        bb_pre_impl = getattr(bb, 'pretrained', None)
        bb_w = getattr(bb, 'weights_path', None)
        bb_out_type = getattr(bb, 'out_type', None)
        bb_out_index = getattr(bb, 'out_index', None)
        logger.info('[train_debug] backbone_impl=%s pretrained_flag=%s weights_path=%s out_type=%s out_index=%s',
                    bb_impl, str(bb_pre_impl), str(bb_w), str(bb_out_type), str(bb_out_index))
    if head is not None:
        logger.info('[train_debug] head_param_count=%d head_trainable=%d', head_total, head_train)

def train_single_epoch_krn(epoch, cfg, model, data_loader, optimizer,
                       writer, device, styleAugmentor=None, scaler=None):
    training_time_meter = AverageMeter('ms')
    loss_x_meter = AverageMeter('-')
    loss_y_meter = AverageMeter('-')
    loss_hm_meter = AverageMeter('-')
    pos_loss_meter = AverageMeter('-')
    neg_loss_meter = AverageMeter('-')
    rmse_meter   = AverageMeter('pix')
    inside_meter = AverageMeter('%')
    naninf_meter = AverageMeter('cnt')

    # switch to train mode
    model.train()
    _debug_print_startup(cfg, model, optimizer)

    # Current learning rate (report the last group's lr)
    lr = optimizer.param_groups[-1]['lr']

    # Loop through dataloader
    for idx, (images, target) in enumerate(data_loader):
        start = time.time()
        B     = images.shape[0]

        # Debug (uncomment)
        # imshow(images[0])
        # scatter_keypoints(images[0], target[0,0], target[0,1], True)

        # To device
        images = images.to(device)
        target = target.to(device)

        # Randomize texture?
        if styleAugmentor is not None and random.random() < cfg.texture_ratio:
            images = styleAugmentor(images)
            # imshow(images[0].cpu())

        # compute output
        if scaler is not None and cfg.use_cuda:
            # Use mixed-precision
            with autocast():
                loss, summary = model(images, target)
        else:
            loss, summary = model(images, target)

        if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap':
            if getattr(model, 'last_pred_heatmaps', None) is None:
                raise RuntimeError('Heatmap head must set last_pred_heatmaps')
            if getattr(model, 'last_gt_heatmaps', None) is None:
                raise RuntimeError('Heatmap head must set last_gt_heatmaps')
            global _TRAIN_DEBUG_BATCH_ONCE
            if not _TRAIN_DEBUG_BATCH_ONCE:
                _TRAIN_DEBUG_BATCH_ONCE = True
                pred_hm = model.last_pred_heatmaps
                gt_hm = model.last_gt_heatmaps
                logits = getattr(model, 'last_pred_heatmaps_logits', None)
                logger.info('[train_debug] output_logits_shape=%s', tuple(logits.shape) if logits is not None else None)
                logger.info('[train_debug] pred_heatmap_shape=%s gt_heatmap_shape=%s',
                            tuple(pred_hm.shape) if pred_hm is not None else None,
                            tuple(gt_hm.shape) if gt_hm is not None else None)
                logger.info('[train_debug] loss_fn=%s heatmap_target=%s',
                            getattr(model, 'heatmap_loss_type', 'unknown'),
                            'yes' if gt_hm is not None else 'no')

        if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap':
            if ('loss_hm' not in summary) or ('pos_loss' not in summary) or ('neg_loss' not in summary):
                raise RuntimeError('Heatmap head must return loss_hm/pos_loss/neg_loss in summary')
            if ('loss_x' in summary) and ('loss_y' in summary):
                if abs(float(summary['loss_x']) - float(summary['loss_hm'])) > 1e-6:
                    raise RuntimeError('Heatmap head is using coordinate regression loss_x/loss_y')

        if not loss.isfinite().all():
            naninf_meter.update(1.0, B)
            continue

        # Zero gradient
        optimizer.zero_grad(set_to_none=True)

        # Compute & update gradient
        if scaler is not None:
            # Use mixed-precision
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap' and epoch == 1 and idx < 10:
                head_g2 = 0.0
                bb_g2 = 0.0
                for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g2 = float(p.grad.detach().float().pow(2).sum().cpu())
                    if n.startswith('backbone.'):
                        bb_g2 += g2
                    else:
                        head_g2 += g2
                head_g = head_g2 ** 0.5
                bb_g = bb_g2 ** 0.5
                pg_sizes = [len(pg.get('params', [])) for pg in optimizer.param_groups]
                s = summary if isinstance(summary, dict) else {}
                print(
                    f'iter={idx:04d} loss_hm={s.get("loss_hm", s.get("loss_x", float("nan"))):.6f} '
                    f'pred_logits(min/max/mean/std)='
                    f'{s.get("pred_logits_min", float("nan")):.4f}/'
                    f'{s.get("pred_logits_max", float("nan")):.4f}/'
                    f'{s.get("pred_logits_mean", float("nan")):.4f}/'
                    f'{s.get("pred_logits_std", float("nan")):.4f} '
                    f'pred_hm(min/max/mean/std)='
                    f'{s.get("pred_hm_min", float("nan")):.4f}/'
                    f'{s.get("pred_hm_max", float("nan")):.4f}/'
                    f'{s.get("pred_hm_mean", float("nan")):.4f}/'
                    f'{s.get("pred_hm_std", float("nan")):.4f} '
                    f'gt_hm(min/max/mean/std)='
                    f'{s.get("gt_hm_min", float("nan")):.4f}/'
                    f'{s.get("gt_hm_max", float("nan")):.4f}/'
                    f'{s.get("gt_hm_mean", float("nan")):.4f}/'
                    f'{s.get("gt_hm_std", float("nan")):.4f} '
                    f'head_grad_norm={head_g:.6f} backbone_grad_norm={bb_g:.6f} opt_groups={pg_sizes}'
                )
            clip_grad_norm_(model.parameters(), float(getattr(cfg, 'grad_clip_norm', 1.0)))
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()
        else:
            loss.backward()
            if cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap' and epoch == 1 and idx < 10:
                head_g2 = 0.0
                bb_g2 = 0.0
                for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g2 = float(p.grad.detach().float().pow(2).sum().cpu())
                    if n.startswith('backbone.'):
                        bb_g2 += g2
                    else:
                        head_g2 += g2
                head_g = head_g2 ** 0.5
                bb_g = bb_g2 ** 0.5
                pg_sizes = [len(pg.get('params', [])) for pg in optimizer.param_groups]
                s = summary if isinstance(summary, dict) else {}
                print(
                    f'iter={idx:04d} loss_hm={s.get("loss_hm", s.get("loss_x", float("nan"))):.6f} '
                    f'pred_logits(min/max/mean/std)='
                    f'{s.get("pred_logits_min", float("nan")):.4f}/'
                    f'{s.get("pred_logits_max", float("nan")):.4f}/'
                    f'{s.get("pred_logits_mean", float("nan")):.4f}/'
                    f'{s.get("pred_logits_std", float("nan")):.4f} '
                    f'pred_hm(min/max/mean/std)='
                    f'{s.get("pred_hm_min", float("nan")):.4f}/'
                    f'{s.get("pred_hm_max", float("nan")):.4f}/'
                    f'{s.get("pred_hm_mean", float("nan")):.4f}/'
                    f'{s.get("pred_hm_std", float("nan")):.4f} '
                    f'gt_hm(min/max/mean/std)='
                    f'{s.get("gt_hm_min", float("nan")):.4f}/'
                    f'{s.get("gt_hm_max", float("nan")):.4f}/'
                    f'{s.get("gt_hm_mean", float("nan")):.4f}/'
                    f'{s.get("gt_hm_std", float("nan")):.4f} '
                    f'head_grad_norm={head_g:.6f} backbone_grad_norm={bb_g:.6f} opt_groups={pg_sizes}'
                )
            clip_grad_norm_(model.parameters(), float(getattr(cfg, 'grad_clip_norm', 1.0)))
            optimizer.step()

        # measure elapsed time & record loss
        training_time_meter.update((time.time() - start)*1000, B)
        is_heatmap = (cfg.model_name == 'krn' and getattr(cfg, 'krn_head', 'direct') == 'heatmap')
        if not is_heatmap:
            loss_x_meter.update(summary.get('loss_x', float(loss.detach().cpu())), B)
            loss_y_meter.update(summary.get('loss_y', float(loss.detach().cpu())), B)
        if 'loss_hm' in summary:
            loss_hm_meter.update(summary['loss_hm'], B)
        if 'pos_loss' in summary:
            pos_loss_meter.update(summary['pos_loss'], B)
        if 'neg_loss' in summary:
            neg_loss_meter.update(summary['neg_loss'], B)
        if 'rmse_px' in summary:
            rmse_meter.update(summary['rmse_px'], B)
        if 'inside_pct' in summary:
            inside_meter.update(summary['inside_pct'], B)

        # report training progress
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=len(data_loader),
                        time=training_time_meter, is_train=True,
                        loss_x=None if is_heatmap else loss_x_meter,
                        loss_y=None if is_heatmap else loss_y_meter,
                        loss_hm=loss_hm_meter if loss_hm_meter.count > 0 else None,
                        pos_loss=pos_loss_meter if pos_loss_meter.count > 0 else None,
                        neg_loss=neg_loss_meter if neg_loss_meter.count > 0 else None,
                        rmse=rmse_meter, inside=inside_meter)

    # log to tensorboard
    if writer is not None:
        if not is_heatmap:
            writer.add_scalar('train/loss_x', loss_x_meter.avg, epoch)
            writer.add_scalar('train/loss_y', loss_y_meter.avg, epoch)
        if loss_hm_meter.count > 0:
            writer.add_scalar('train/loss_hm', loss_hm_meter.avg, epoch)
        if pos_loss_meter.count > 0:
            writer.add_scalar('train/pos_loss', pos_loss_meter.avg, epoch)
        if neg_loss_meter.count > 0:
            writer.add_scalar('train/neg_loss', neg_loss_meter.avg, epoch)
        if rmse_meter.count > 0:
            writer.add_scalar('train/rmse_px', rmse_meter.avg, epoch)
        if inside_meter.count > 0:
            writer.add_scalar('train/inside_pct', inside_meter.avg, epoch)
        writer.add_scalar('train/naninf_cnt', naninf_meter.sum, epoch)

    return {
        'loss_hm': loss_hm_meter.avg if loss_hm_meter.count > 0 else float('nan'),
        'pos_loss': pos_loss_meter.avg if pos_loss_meter.count > 0 else float('nan'),
        'neg_loss': neg_loss_meter.avg if neg_loss_meter.count > 0 else float('nan'),
        'rmse_px': rmse_meter.avg if rmse_meter.count > 0 else float('nan'),
        'inside_pct': inside_meter.avg if inside_meter.count > 0 else float('nan'),
    }

def train_single_epoch_spn(epoch, cfg, model, data_loader, optimizer,
                       writer, device, styleAugmentor=None, scaler=None):
    training_time_meter = AverageMeter('ms')
    loss_class_meter  = AverageMeter('-')
    loss_weight_meter = AverageMeter('-')

    # switch to train mode
    model.train()

    # Current learning rate
    for pg in optimizer.param_groups:
        lr = pg['lr']

    # Loop through dataloader
    for idx, (images, yClasses, yWeights) in enumerate(data_loader):
        start = time.time()
        B     = images.shape[0]

        # Debug (uncomment)
        # imshow(images[0])

        # To device
        images   = images.to(device)
        yClasses = yClasses.to(device)
        yWeights = yWeights.to(device)

        # Randomize texture?
        if styleAugmentor is not None and random.random() < cfg.texture_ratio:
            images = styleAugmentor(images)
            # imshow(images[0].cpu())

        # compute output
        if scaler is not None and cfg.use_cuda:
            # Use mixed-precision
            with autocast():
                classes, weights = model(images)

                # Attitude classification / Relative attitude loss
                loss_class   = softmax_cross_entropy_with_logits(classes, yClasses, reduction='mean')
                loss_regress = softmax_cross_entropy_with_logits(weights, yWeights, reduction='mean')

                # Fina loss
                loss = loss_class + 10.0 * loss_regress
        else:
            classes, weights = model(images)

            # Attitude classification / Relative attitude loss
            loss_class   = softmax_cross_entropy_with_logits(classes, yClasses, reduction='mean')
            loss_regress = softmax_cross_entropy_with_logits(weights, yWeights, reduction='mean')

            # Fina loss
            loss = loss_class + 10.0 * loss_regress

        # Zero gradient
        optimizer.zero_grad(set_to_none=True)

        # Compute & update gradient
        if scaler is not None:
            # Use mixed-precision
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            clip_grad_value_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()
        else:
            loss.backward()
            clip_grad_value_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()

        # measure elapsed time & loss
        training_time_meter.update((time.time() - start)*1000, B)
        loss_class_meter.update(float(loss_class.detach().cpu()), B)
        loss_weight_meter.update(float(loss_regress.detach().cpu()), B)

        # report training progress
        report_progress(epoch=epoch, lr=lr, epoch_iter=idx+1, epoch_size=len(data_loader),
                        time=training_time_meter, is_train=True, loss_c=loss_class_meter, loss_r=loss_weight_meter)

    # log to tensorboard
    if writer is not None:
        writer.add_scalar('train/loss_c', loss_class_meter.avg, epoch)
        writer.add_scalar('train/loss_r', loss_weight_meter.avg, epoch)
