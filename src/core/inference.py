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
import os.path as osp
from scipy.spatial.transform import Rotation as R

import torch
import numpy as np

from src.utils.utils     import pnp, pnp_ransac, weighted_mean_quaternion, \
                                AverageMeter, report_progress, quat2dcm
from src.utils.utils     import project_keypoints
from src.utils.computePositionSPN import compute_position_spn
from src.utils.metrics   import *
from src.utils.visualize import imshow, plot_2D_bbox, scatter_keypoints
from src.utils.heatmaps import keypoint_rmse_pixels, inside_image_percentage
from src.utils.heatmaps import gaussian_heatmaps_from_keypoints
from src.utils.heatmaps import per_keypoint_error_pixels, heatmap_peaks, heatmap_entropy, heatmap_peak_to_second_ratio, per_keypoint_pck, collapsed_keypoint_distance, keypoint_spread_min_px, heatmap_topk_peaks
from src.utils.heatmap_pipeline import reprojection_errors_px, pose_valid

logger = logging.getLogger("Testing")

def _nanmedian(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(np.median(arr))

def _nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(arr.mean())

class _ScalarPerf(object):
    def __init__(self, avg, unit='-'):
        self.avg = float(avg)
        self.unit = unit

def _geom_valid_mask_from_pose(gt_kp_norm, bbox_xyxy, q_gt, t_gt, corners3D, cameraMatrix, distCoeffs, input_shape, heatmap_size, z_min):
    gt = gt_kp_norm.detach()
    K = int(gt.shape[-1])
    xs = gt[0, :].to(dtype=torch.float32)
    ys = gt[1, :].to(dtype=torch.float32)
    valid = torch.isfinite(xs) & torch.isfinite(ys)

    H, W = int(input_shape[0]), int(input_shape[1])
    valid = valid & (xs >= 0.0) & (xs <= 1.0) & (ys >= 0.0) & (ys <= 1.0)
    x_img = xs * max(float(W - 1), 1.0)
    y_img = ys * max(float(H - 1), 1.0)
    valid = valid & torch.isfinite(x_img) & torch.isfinite(y_img) & (x_img >= 0.0) & (x_img <= float(W - 1)) & (y_img >= 0.0) & (y_img <= float(H - 1))

    Hm, Wm = int(heatmap_size[0]), int(heatmap_size[1])
    x_hm = xs * max(float(Wm - 1), 1.0)
    y_hm = ys * max(float(Hm - 1), 1.0)
    valid = valid & torch.isfinite(x_hm) & torch.isfinite(y_hm) & (x_hm >= 0.0) & (x_hm <= float(Wm - 1)) & (y_hm >= 0.0) & (y_hm <= float(Hm - 1))

    z_thr = float(z_min) if z_min is not None else 0.0
    if z_thr > 0.0:
        q = np.asarray(q_gt, dtype=np.float64).reshape(4)
        t = np.asarray(t_gt, dtype=np.float64).reshape(3)
        p3d = np.asarray(corners3D, dtype=np.float64).reshape(-1, 3)
        if p3d.shape[0] != K:
            p3d = p3d[:K, :]
        keypoints = np.transpose(p3d)  # (3,K)
        keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]), dtype=np.float64)))
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(t, 1)))
        xyz = np.dot(pose_mat, keypoints)
        zc = xyz[2, :]
        z_ok = torch.from_numpy((np.isfinite(zc) & (zc > z_thr)).astype(np.bool_))
        valid = valid & z_ok.to(device=valid.device)

        pts2d = None
        try:
            pts2d = project_keypoints(q, t, cameraMatrix, distCoeffs, p3d)
        except Exception:
            pts2d = None
        if pts2d is not None:
            pts2d = np.asarray(pts2d, dtype=np.float64)
            if pts2d.shape[0] != 2:
                pts2d = pts2d.reshape(2, -1)
            bb = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
            dx = max(float(bb[1] - bb[0]), 1.0)
            dy = max(float(bb[3] - bb[2]), 1.0)
            xn = (pts2d[0, :K] - float(bb[0])) / dx
            yn = (pts2d[1, :K] - float(bb[2])) / dy
            ok2 = np.isfinite(xn) & np.isfinite(yn) & (xn >= 0.0) & (xn <= 1.0) & (yn >= 0.0) & (yn <= 1.0)
            valid = valid & torch.from_numpy(ok2.astype(np.bool_)).to(device=valid.device)

    return valid.to(dtype=torch.bool)

def valid_krn(epoch, cfg, model, data_loader, cameraMatrix, distCoeffs, corners3D, writer, device, qClass=None):
    ''' Validate KRN model '''

    # Initialize trackers
    test_time_meter = AverageMeter('ms')
    err_q_meter     = AverageMeter('deg')
    err_t_meter     = AverageMeter('m')
    speed_meter     = AverageMeter('-')
    speed_meter_th  = AverageMeter('-')
    acc_meter       = AverageMeter('%')
    rmse_meter      = AverageMeter('pix')
    rmse_norm_input_meter = AverageMeter('-')
    rmse_norm_bbox_meter = AverageMeter('-')
    inside_meter    = AverageMeter('%')
    valid_kpt_meter = AverageMeter('%')
    pnp_fail_meter  = AverageMeter('cnt')
    pnp_fail_pct_meter = AverageMeter('%')
    pose_valid_meter = AverageMeter('cnt')
    reproj_med_meter = AverageMeter('pix')
    hm_peak_mean_meter = AverageMeter('-')
    hm_peak_std_meter = AverageMeter('-')
    hm_entropy_meter = AverageMeter('-')
    hm_entropy_std_meter = AverageMeter('-')
    hm_entropy_beta_meter = AverageMeter('-')
    hm_entropy_beta_std_meter = AverageMeter('-')
    collapsed_kpt_dist_meter = AverageMeter('pix')
    collapsed_kpt_spread_min_meter = AverageMeter('pix')
    per_kpt_rmse_meters = [AverageMeter('pix') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    per_kpt_pck5_meters = [AverageMeter('%') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    per_kpt_pck10_meters = [AverageMeter('%') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    per_kpt_pck05bbox_meters = [AverageMeter('%') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    per_kpt_pck10bbox_meters = [AverageMeter('%') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    pck05bbox_meter = AverageMeter('%')
    pck10bbox_meter = AverageMeter('%')
    per_kpt_peak_meters = [AverageMeter('-') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    per_kpt_peak2_meters = [AverageMeter('-') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    per_kpt_peak_ratio_meters = [AverageMeter('-') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    per_kpt_top1_to_gt_dist_hm_px_meters = [AverageMeter('pix') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]
    per_kpt_top2_to_gt_dist_hm_px_meters = [AverageMeter('pix') for _ in range(int(getattr(cfg, 'num_keypoints', 11)))]

    ransac_fail_meter = AverageMeter('cnt')
    ransac_inlier_meter = AverageMeter('cnt')
    err_q_ransac_meter = AverageMeter('deg')
    err_t_ransac_meter = AverageMeter('m')

    err_q_all = []
    err_t_all = []
    speed_raw_all = []
    speed_mod_all = []
    keypoint_rmse_all = []
    keypoint_rmse_norm_input_all = []
    keypoint_rmse_norm_bbox_all = []
    pnp_ok_err_t_all = []
    pnp_ok_err_q_all = []
    ransac_ok_err_t_all = []
    ransac_ok_err_q_all = []
    ransac_inlier_all = []
    reproj_med_all = []
    pose_valid_all = []

    nK = int(getattr(cfg, 'num_keypoints', 11))
    hard_total_valid = np.zeros((nK,), dtype=np.int64)
    hard_invalid = np.zeros((nK,), dtype=np.int64)
    hard_err_gt10 = np.zeros((nK,), dtype=np.int64)
    hard_err_gt20 = np.zeros((nK,), dtype=np.int64)
    hard_near_border = np.zeros((nK,), dtype=np.int64)
    hard_near_border_err_gt10 = np.zeros((nK,), dtype=np.int64)

    ctx_all_cnt = np.zeros((nK,), dtype=np.int64)
    ctx_hard_cnt = np.zeros((nK,), dtype=np.int64)
    ctx_all_bbox_area_sum = np.zeros((nK,), dtype=np.float64)
    ctx_hard_bbox_area_sum = np.zeros((nK,), dtype=np.float64)
    ctx_all_bbox_min_sum = np.zeros((nK,), dtype=np.float64)
    ctx_hard_bbox_min_sum = np.zeros((nK,), dtype=np.float64)
    ctx_all_tnorm_sum = np.zeros((nK,), dtype=np.float64)
    ctx_hard_tnorm_sum = np.zeros((nK,), dtype=np.float64)
    ctx_all_rotdeg_sum = np.zeros((nK,), dtype=np.float64)
    ctx_hard_rotdeg_sum = np.zeros((nK,), dtype=np.float64)
    ctx_all_brightness_sum = np.zeros((nK,), dtype=np.float64)
    ctx_hard_brightness_sum = np.zeros((nK,), dtype=np.float64)
    ctx_all_borderpx_sum = np.zeros((nK,), dtype=np.float64)
    ctx_hard_borderpx_sum = np.zeros((nK,), dtype=np.float64)

    ransac_inlier_sel_cnt = np.zeros((nK,), dtype=np.int64)
    ransac_inlier_cand_cnt = np.zeros((nK,), dtype=np.int64)

    debug_save_n = int(getattr(cfg, 'debug_save_n', 20) or 20)
    worst_k = debug_save_n
    best_k = debug_save_n
    worst = []
    best = []

    # switch to eval mode
    model.eval()
    has_keypts = False
    debug_dir = osp.join(cfg.logdir, 'ranked', f'epoch_{int(epoch):03d}')

    # Loop through dataloader
    for idx, batch in enumerate(data_loader):
        start = time.time()
        if len(batch) == 4:
            images, bbox, q_gt, t_gt = batch
            keypts_gt = None
        else:
            images, bbox, keypts_gt, q_gt, t_gt = batch
            has_keypts = True

        B     = images.shape[0]

        # Debug (uncomment)
        # imshow(images[0])
        # print(bbox[0])

        # To device
        images = images.to(device)
        with torch.no_grad():
            x_pr, y_pr = model(images)

        # Debug (uncomment)
        # scatter_keypoints(images[0].cpu(), x_pr[0].cpu(), y_pr[0].cpu(), normalized=True)

        for b in range(B):
            sample_id = idx * B + b
            pred_kp_norm = torch.stack([x_pr[b].to(device), y_pr[b].to(device)], dim=0)  # (2,K)
            geom_valid = None
            if keypts_gt is not None:
                has_keypts = True
                gt_kp_norm = keypts_gt[b].to(device)  # (2,K)
                z_min = float(getattr(cfg, 'z_min', 0.0) or 0.0)
                geom_valid = _geom_valid_mask_from_pose(
                    gt_kp_norm,
                    bbox[b].detach().cpu().numpy(),
                    q_gt[b].detach().cpu().numpy(),
                    t_gt[b].detach().cpu().numpy(),
                    corners3D,
                    cameraMatrix,
                    distCoeffs,
                    input_shape=getattr(cfg, 'input_shape', (224, 224)),
                    heatmap_size=getattr(cfg, 'heatmap_size', (56, 56)),
                    z_min=z_min,
                )

            # Post-processing
            pnp_failed = False
            try:
                if geom_valid is not None:
                    q_pr, t_pr, _ = _keypts_to_pose_masked(x_pr[b], y_pr[b], bbox[b], corners3D, cameraMatrix, distCoeffs, geom_valid)
                else:
                    q_pr, t_pr = _keypts_to_pose(x_pr[b], y_pr[b], bbox[b], corners3D, cameraMatrix, distCoeffs)
            except Exception:
                pnp_failed = True
                q_pr = np.array([1, 0, 0, 0], dtype=np.float32)
                t_pr = np.array([0, 0, 0], dtype=np.float32)

            reproj_med = float('nan')
            pose_ok = False
            if not pnp_failed:
                try:
                    vm_np = geom_valid.detach().cpu().numpy() if geom_valid is not None else None
                    reproj_err = reprojection_errors_px(pred_kp_norm.detach().cpu(), bbox[b].detach().cpu().numpy(), q_pr, t_pr, corners3D, cameraMatrix, distCoeffs)
                    if vm_np is not None:
                        reproj_med = float(np.median(np.asarray(reproj_err)[vm_np])) if np.asarray(reproj_err)[vm_np].size > 0 else float('inf')
                    else:
                        reproj_med = float(np.median(np.asarray(reproj_err)))
                    pose_ok = pose_valid(
                        q_pr, t_pr, reproj_err,
                        valid_mask=vm_np,
                        min_valid_k=4,
                        reproj_median_thr_px=float(getattr(cfg, 'pose_reproj_thr_px', 10.0)),
                        t_norm_range=(float(getattr(cfg, 'pose_t_min', 0.5)), float(getattr(cfg, 'pose_t_max', 20.0))),
                    )
                except Exception:
                    pose_ok = False
                    reproj_med = float('nan')

            ransac_failed = False
            inlier_idx = np.zeros((0,), dtype=np.int64)
            inlier_cands = np.arange(nK, dtype=np.int64)
            try:
                if geom_valid is not None:
                    q_r, t_r, inlier_idx, inlier_cands = _keypts_to_pose_ransac_masked(
                        x_pr[b], y_pr[b], bbox[b], corners3D, cameraMatrix, distCoeffs,
                        valid_mask=geom_valid,
                        reproj_thr_px=float(getattr(cfg, 'ransac_reproj_thr_px', 8.0)),
                    )
                else:
                    q_r, t_r, inlier_idx = _keypts_to_pose_ransac(
                        x_pr[b], y_pr[b], bbox[b], corners3D, cameraMatrix, distCoeffs,
                        reproj_thr_px=float(getattr(cfg, 'ransac_reproj_thr_px', 8.0)),
                    )
            except Exception:
                ransac_failed = True
                q_r = np.array([1, 0, 0, 0], dtype=np.float32)
                t_r = np.array([0, 0, 0], dtype=np.float32)
                inlier_idx = np.zeros((0,), dtype=np.int64)
                inlier_cands = np.zeros((0,), dtype=np.int64)
            inl = int(np.asarray(inlier_idx).reshape(-1).shape[0])

            # Ground-truth
            q_gt_i = q_gt[b].numpy()
            t_gt_i = t_gt[b].numpy()

            # Metrics
            if pnp_failed:
                err_q = float('nan')
                err_t = float('nan')
                speed_raw = float('nan')
                speed_mod = float('nan')
                acc = float('nan')
            else:
                err_q = error_orientation(q_pr, q_gt_i) # [deg]
                err_t = error_translation(t_pr, t_gt_i)
                speed_raw, acc = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=False)
                speed_mod, _   = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=True,
                        rotThresh=0.169, posThresh=0.002173)

            if ransac_failed:
                err_q_r = float('nan')
                err_t_r = float('nan')
            else:
                err_q_r = error_orientation(q_r, q_gt_i)
                err_t_r = error_translation(t_r, t_gt_i)

            if keypts_gt is not None:
                gt = gt_kp_norm.unsqueeze(0)  # (1,2,K)
                pr = pred_kp_norm.unsqueeze(0)  # (1,2,K)
                vm = geom_valid if geom_valid is not None else (torch.isfinite(gt[:, 0, :]) & torch.isfinite(gt[:, 1, :]) & (gt[:, 0, :] >= 0) & (gt[:, 0, :] <= 1) & (gt[:, 1, :] >= 0) & (gt[:, 1, :] <= 1)).squeeze(0)
                rmse = keypoint_rmse_pixels(pr, gt, image_size=cfg.input_shape, valid_mask=vm.unsqueeze(0))
                inside = inside_image_percentage(pr, vm.unsqueeze(0))
                valid_pct = vm.to(dtype=torch.float32).mean() * 100.0
                per_err = per_keypoint_error_pixels(pr, gt, image_size=cfg.input_shape).squeeze(0)
                rmse_v = float(rmse.detach().cpu())
                keypoint_rmse_all.append(rmse_v)

                H, W = int(cfg.input_shape[0]), int(cfg.input_shape[1])
                xs_gt_px = (gt_kp_norm[0].detach().cpu().numpy() * max(float(W - 1), 1.0))
                ys_gt_px = (gt_kp_norm[1].detach().cpu().numpy() * max(float(H - 1), 1.0))
                border_dist = np.minimum.reduce([xs_gt_px, ys_gt_px, (W - 1) - xs_gt_px, (H - 1) - ys_gt_px])

                bb = bbox[b].detach().cpu().numpy().reshape(4)
                bw = max(float(bb[1] - bb[0]), 1.0)
                bh = max(float(bb[3] - bb[2]), 1.0)
                bbox_area = float(bw * bh)
                bbox_min = float(min(bw, bh))
                bbox_diag = float(np.sqrt(bw * bw + bh * bh))
                bbox_diag = max(bbox_diag, 1.0)

                rmse_norm_input = rmse_v / max(float(max(H, W) - 1), 1.0)
                rmse_norm_bbox = rmse_v / bbox_diag

                keypoint_rmse_norm_input_all.append(float(rmse_norm_input))
                keypoint_rmse_norm_bbox_all.append(float(rmse_norm_bbox))

                vm_f = vm.to(dtype=torch.float32)
                denom_k = float(torch.clamp(vm_f.sum(), min=1.0).detach().cpu())
                p05 = float((((per_err <= (0.05 * bbox_diag)).to(dtype=torch.float32) * vm_f).sum() / denom_k).detach().cpu()) * 100.0
                p10 = float((((per_err <= (0.10 * bbox_diag)).to(dtype=torch.float32) * vm_f).sum() / denom_k).detach().cpu()) * 100.0

                tnorm = float(np.linalg.norm(np.asarray(t_gt_i, dtype=np.float64).reshape(3)))
                rotdeg = float('nan')
                try:
                    q = np.asarray(q_gt_i, dtype=np.float64).reshape(4)
                    rr = R.from_quat(np.array([q[1], q[2], q[3], q[0]], dtype=np.float64))
                    rotdeg = float(rr.magnitude() * 180.0 / np.pi)
                except Exception:
                    rotdeg = float('nan')

                img_cpu = images[b].detach().cpu()
                gray = (0.2989 * img_cpu[0] + 0.5870 * img_cpu[1] + 0.1140 * img_cpu[2]).to(dtype=torch.float32)
                pad = int(getattr(cfg, 'brightness_patch', 3) or 3)
                for k in range(min(int(per_err.shape[0]), nK)):
                    if not bool(vm[k].item()):
                        hard_invalid[k] += 1
                        continue

                    hard_total_valid[k] += 1
                    ek = float(per_err[k].detach().cpu())

                    xi = int(round(float(xs_gt_px[k])))
                    yi = int(round(float(ys_gt_px[k])))
                    x0 = max(0, xi - pad)
                    x1 = min(W - 1, xi + pad)
                    y0 = max(0, yi - pad)
                    y1 = min(H - 1, yi + pad)
                    patch = gray[y0 : y1 + 1, x0 : x1 + 1]
                    bright = float(patch.mean().item()) if patch.numel() > 0 else float('nan')

                    ctx_all_cnt[k] += 1
                    ctx_all_bbox_area_sum[k] += bbox_area
                    ctx_all_bbox_min_sum[k] += bbox_min
                    if np.isfinite(tnorm):
                        ctx_all_tnorm_sum[k] += tnorm
                    if np.isfinite(rotdeg):
                        ctx_all_rotdeg_sum[k] += rotdeg
                    if np.isfinite(bright):
                        ctx_all_brightness_sum[k] += bright
                    if np.isfinite(border_dist[k]):
                        ctx_all_borderpx_sum[k] += float(border_dist[k])

                    if ek > 10.0:
                        hard_err_gt10[k] += 1
                    if ek > 20.0:
                        hard_err_gt20[k] += 1
                        ctx_hard_cnt[k] += 1
                        ctx_hard_bbox_area_sum[k] += bbox_area
                        ctx_hard_bbox_min_sum[k] += bbox_min
                        if np.isfinite(tnorm):
                            ctx_hard_tnorm_sum[k] += tnorm
                        if np.isfinite(rotdeg):
                            ctx_hard_rotdeg_sum[k] += rotdeg
                        if np.isfinite(bright):
                            ctx_hard_brightness_sum[k] += bright
                        if np.isfinite(border_dist[k]):
                            ctx_hard_borderpx_sum[k] += float(border_dist[k])

                    if np.isfinite(border_dist[k]) and float(border_dist[k]) <= 5.0:
                        hard_near_border[k] += 1
                        if ek > 10.0:
                            hard_near_border_err_gt10[k] += 1


                for k in range(min(per_err.shape[0], len(per_kpt_rmse_meters))):
                    if bool(vm[k].item()):
                        per_kpt_rmse_meters[k].update(float(per_err[k].detach().cpu()), 1)
                        per_kpt_pck5_meters[k].update(float((per_err[k] <= 5.0).to(dtype=torch.float32).detach().cpu()) * 100.0, 1)
                        per_kpt_pck10_meters[k].update(float((per_err[k] <= 10.0).to(dtype=torch.float32).detach().cpu()) * 100.0, 1)
                        per_kpt_pck05bbox_meters[k].update(float((per_err[k] <= (0.05 * bbox_diag)).to(dtype=torch.float32).detach().cpu()) * 100.0, 1)
                        per_kpt_pck10bbox_meters[k].update(float((per_err[k] <= (0.10 * bbox_diag)).to(dtype=torch.float32).detach().cpu()) * 100.0, 1)
                coll_dist = collapsed_keypoint_distance(pr, image_size=cfg.input_shape, valid_mask=vm.unsqueeze(0))
                spread_min = keypoint_spread_min_px(pr, image_size=cfg.input_shape, valid_mask=vm.unsqueeze(0))
                collapsed_kpt_dist_meter.update(float(coll_dist.detach().cpu()), 1)
                collapsed_kpt_spread_min_meter.update(float(spread_min.detach().cpu()), 1)

                pred_hm = None
                pred_logits = None
                if hasattr(model, 'last_pred_heatmaps') and model.last_pred_heatmaps is not None:
                    try:
                        pred_hm = model.last_pred_heatmaps[b].detach()
                    except Exception:
                        pred_hm = None
                if hasattr(model, 'last_pred_heatmaps_logits') and model.last_pred_heatmaps_logits is not None:
                    try:
                        pred_logits = model.last_pred_heatmaps_logits[b].detach()
                    except Exception:
                        pred_logits = None
                if pred_hm is not None:
                    coords2, vals2 = heatmap_topk_peaks(pred_hm.unsqueeze(0), k=2)
                    coords2 = coords2.squeeze(0)
                    vals2 = vals2.squeeze(0)
                    peaks = vals2[:, 0]
                    peaks2 = vals2[:, 1] if vals2.shape[-1] > 1 else torch.zeros_like(peaks)
                    ratios = peaks / peaks2.clamp(min=1e-12)
                    ent_src = pred_logits if pred_logits is not None else pred_hm
                    ent = heatmap_entropy(ent_src.unsqueeze(0), beta=1.0).squeeze(0)  # (K,)
                    ent_beta = heatmap_entropy(ent_src.unsqueeze(0), beta=float(getattr(cfg, 'heatmap_beta', 100.0))).squeeze(0)  # (K,)
                    m = vm.to(dtype=peaks.dtype)
                    denom = float(torch.clamp(m.sum(), min=1.0).detach().cpu())
                    peak_mean = float(((peaks * m).sum() / denom).detach().cpu())
                    peak_std = float(torch.sqrt(torch.clamp(((peaks - peak_mean) ** 2) * m, min=0.0).sum() / denom).detach().cpu())
                    ent_mean = float(((ent * m).sum() / denom).detach().cpu())
                    ent_std = float(torch.sqrt(torch.clamp(((ent - ent_mean) ** 2) * m, min=0.0).sum() / denom).detach().cpu())
                    ent_beta_mean = float(((ent_beta * m).sum() / denom).detach().cpu())
                    ent_beta_std = float(torch.sqrt(torch.clamp(((ent_beta - ent_beta_mean) ** 2) * m, min=0.0).sum() / denom).detach().cpu())
                    hm_peak_mean_meter.update(peak_mean, 1)
                    hm_peak_std_meter.update(peak_std, 1)
                    hm_entropy_meter.update(ent_mean, 1)
                    hm_entropy_std_meter.update(ent_std, 1)
                    hm_entropy_beta_meter.update(ent_beta_mean, 1)
                    hm_entropy_beta_std_meter.update(ent_beta_std, 1)
                    if keypts_gt is not None:
                        Hh, Wh = int(pred_hm.shape[-2]), int(pred_hm.shape[-1])
                        gx = gt_kp_norm[0].to(device=coords2.device, dtype=coords2.dtype)
                        gy = gt_kp_norm[1].to(device=coords2.device, dtype=coords2.dtype)
                        top1 = coords2[:, :, 0]
                        top2 = coords2[:, :, 1] if coords2.shape[-1] > 1 else coords2[:, :, 0]
                        d1 = torch.sqrt(
                            ((top1[:, 0] - gx) * max(float(Wh - 1), 1.0)) ** 2 + ((top1[:, 1] - gy) * max(float(Hh - 1), 1.0)) ** 2
                        )
                        d2 = torch.sqrt(
                            ((top2[:, 0] - gx) * max(float(Wh - 1), 1.0)) ** 2 + ((top2[:, 1] - gy) * max(float(Hh - 1), 1.0)) ** 2
                        )
                    for k in range(min(int(peaks.shape[0]), len(per_kpt_peak_meters))):
                        if bool(vm[k].item()):
                            per_kpt_peak_meters[k].update(float(peaks[k].detach().cpu()), 1)
                            per_kpt_peak2_meters[k].update(float(peaks2[k].detach().cpu()), 1)
                            per_kpt_peak_ratio_meters[k].update(float(ratios[k].detach().cpu()), 1)
                            if keypts_gt is not None:
                                per_kpt_top1_to_gt_dist_hm_px_meters[k].update(float(d1[k].detach().cpu()), 1)
                                per_kpt_top2_to_gt_dist_hm_px_meters[k].update(float(d2[k].detach().cpu()), 1)

            pnp_fail_meter.update(float(1 if pnp_failed else 0), 1)
            pnp_fail_pct_meter.update(float(100.0 if pnp_failed else 0.0), 1)
            pose_valid_meter.update(float(1 if pose_ok else 0), 1)
            if np.isfinite(reproj_med):
                reproj_med_meter.update(float(reproj_med), 1)
                reproj_med_all.append(float(reproj_med))
            pose_valid_all.append(bool(pose_ok))
            if (not pnp_failed) and np.isfinite(err_q) and np.isfinite(err_t):
                err_q_meter.update(float(err_q), 1)
                err_t_meter.update(float(err_t), 1)
                pnp_ok_err_q_all.append(float(err_q))
                pnp_ok_err_t_all.append(float(err_t))
                if np.isfinite(speed_raw):
                    speed_meter.update(float(speed_raw), 1)
                if np.isfinite(speed_mod):
                    speed_meter_th.update(float(speed_mod), 1)
                if np.isfinite(acc):
                    acc_meter.update(float(acc) * 100.0, 1)
            if keypts_gt is not None:
                rmse_meter.update(float(rmse.detach().cpu()), 1)
                rmse_norm_input_meter.update(float(rmse_norm_input), 1)
                rmse_norm_bbox_meter.update(float(rmse_norm_bbox), 1)
                inside_meter.update(float(inside.detach().cpu()), 1)
                valid_kpt_meter.update(float(valid_pct.detach().cpu()), 1)
                pck05bbox_meter.update(float(p05), 1)
                pck10bbox_meter.update(float(p10), 1)

            ransac_fail_meter.update(float(1 if ransac_failed else 0), 1)
            if (not ransac_failed) and np.isfinite(err_q_r) and np.isfinite(err_t_r):
                ransac_inlier_meter.update(float(inl), 1)
                err_q_ransac_meter.update(float(err_q_r), 1)
                err_t_ransac_meter.update(float(err_t_r), 1)
                ransac_ok_err_q_all.append(float(err_q_r))
                ransac_ok_err_t_all.append(float(err_t_r))
                ransac_inlier_all.append(float(inl))

                inlier_idx = np.asarray(inlier_idx, dtype=np.int64).reshape(-1)
                inlier_cands = np.asarray(inlier_cands, dtype=np.int64).reshape(-1)
                for k in inlier_cands:
                    if 0 <= int(k) < nK:
                        ransac_inlier_cand_cnt[int(k)] += 1
                for k in inlier_idx:
                    if 0 <= int(k) < nK:
                        ransac_inlier_sel_cnt[int(k)] += 1

            err_q_all.append(err_q)
            err_t_all.append(err_t)
            speed_raw_all.append(speed_raw)
            speed_mod_all.append(speed_mod)

            if (worst_k > 0 or best_k > 0) and keypts_gt is not None:
                score = float('inf') if (not np.isfinite(rmse_v)) else float(rmse_v)
                item = {
                    'score': score,
                    'sample_id': int(sample_id),
                    'image': images[b].detach().cpu(),
                    'bbox': bbox[b].detach().cpu(),
                    'kp_gt': gt_kp_norm.detach().cpu(),
                    'kp_pr': pred_kp_norm.detach().cpu(),
                    'vm': vm.detach().cpu(),
                    'per_err': per_err.detach().cpu(),
                    'pnp_failed': bool(pnp_failed),
                    'ransac_failed': bool(ransac_failed),
                    'q_pr': np.asarray(q_pr, dtype=np.float32),
                    't_pr': np.asarray(t_pr, dtype=np.float32),
                    'q_r': np.asarray(q_r, dtype=np.float32),
                    't_r': np.asarray(t_r, dtype=np.float32),
                    'inlier_idx': np.asarray(inlier_idx, dtype=np.int64).reshape(-1),
                }
                if hasattr(model, 'last_pred_heatmaps') and model.last_pred_heatmaps is not None:
                    try:
                        item['pred_hm'] = model.last_pred_heatmaps[b].detach().cpu()
                        coords3, vals3 = heatmap_topk_peaks(item['pred_hm'].unsqueeze(0), k=3)
                        item['top3_coords'] = coords3.squeeze(0).detach().cpu()
                        item['top3_vals'] = vals3.squeeze(0).detach().cpu()
                        item['peak_ratio'] = heatmap_peak_to_second_ratio(item['pred_hm'].unsqueeze(0)).squeeze(0).detach().cpu()
                    except Exception:
                        pass
                if worst_k > 0:
                    worst.append(item)
                    if len(worst) > max(worst_k * 4, worst_k):
                        worst.sort(key=lambda d: (float(d['score']) if np.isfinite(d['score']) else float('inf')), reverse=True)
                        worst = worst[:worst_k]
                if best_k > 0:
                    best.append(item)
                    if len(best) > max(best_k * 4, best_k):
                        best.sort(key=lambda d: (float(d['score']) if np.isfinite(d['score']) else float('inf')))
                        best = best[:best_k]

        test_time_meter.update((time.time()-start)*1000, B)

        report_progress(epoch=epoch, lr=np.nan, epoch_iter=idx+1, epoch_size=len(data_loader),
                        time=test_time_meter, is_train=False, eT=err_t_meter, eR=err_q_meter,
                        speed=speed_meter, acc=acc_meter, pnp_fail=pnp_fail_pct_meter)

    if err_t_meter.count == 0:
        for m in (err_q_meter, err_t_meter, speed_meter, speed_meter_th, acc_meter):
            m.val = float('nan')
            m.avg = float('nan')

    # log to Tensorboard
    if writer is not None:
        writer.add_scalar('Valid/err_q [deg]', err_q_meter.avg, epoch)
        writer.add_scalar('Valid/err_t [m]',   err_t_meter.avg, epoch)
        writer.add_scalar('Valid/speed (raw) [-]', speed_meter.avg, epoch)
        writer.add_scalar('Valid/speed (thr) [-]', speed_meter_th.avg, epoch)
        if has_keypts:
            writer.add_scalar('Valid/keypoint_rmse_px', rmse_meter.avg, epoch)
            writer.add_scalar('Valid/keypoint_rmse_px_median', _nanmedian(keypoint_rmse_all), epoch)
            writer.add_scalar('Valid/keypoint_rmse_norm_input', rmse_norm_input_meter.avg, epoch)
            writer.add_scalar('Valid/keypoint_rmse_norm_input_median', _nanmedian(keypoint_rmse_norm_input_all), epoch)
            writer.add_scalar('Valid/keypoint_rmse_norm_bbox', rmse_norm_bbox_meter.avg, epoch)
            writer.add_scalar('Valid/keypoint_rmse_norm_bbox_median', _nanmedian(keypoint_rmse_norm_bbox_all), epoch)
            writer.add_scalar('Valid/pck_0.05_bbox_pct', pck05bbox_meter.avg, epoch)
            writer.add_scalar('Valid/pck_0.10_bbox_pct', pck10bbox_meter.avg, epoch)
            writer.add_scalar('Valid/keypoint_inside_pct', inside_meter.avg, epoch)
            writer.add_scalar('Valid/valid_kpt_pct', valid_kpt_meter.avg, epoch)
            if hm_peak_mean_meter.count > 0:
                writer.add_scalar('Valid/heatmap_peak_mean', hm_peak_mean_meter.avg, epoch)
                writer.add_scalar('Valid/heatmap_peak_std', hm_peak_std_meter.avg, epoch)
                writer.add_scalar('Valid/heatmap_entropy', hm_entropy_meter.avg, epoch)
                writer.add_scalar('Valid/heatmap_entropy_std', hm_entropy_std_meter.avg, epoch)
                writer.add_scalar('Valid/heatmap_entropy_beta', hm_entropy_beta_meter.avg, epoch)
                writer.add_scalar('Valid/heatmap_entropy_beta_std', hm_entropy_beta_std_meter.avg, epoch)
            if collapsed_kpt_dist_meter.count > 0:
                writer.add_scalar('Valid/collapsed_keypoint_distance', collapsed_kpt_dist_meter.avg, epoch)
                writer.add_scalar('Valid/collapsed_keypoint_spread_min_px', collapsed_kpt_spread_min_meter.avg, epoch)
        writer.add_scalar('Valid/pnp_fail_cnt', pnp_fail_meter.sum, epoch)
        writer.add_scalar('Valid/pnp_ok_cnt', float(max(int(pnp_fail_meter.count) - float(pnp_fail_meter.sum), 0.0)), epoch)
        writer.add_scalar('Valid/pnp_fail_pct', (float(pnp_fail_meter.sum) / max(float(pnp_fail_meter.count), 1.0)) * 100.0, epoch)
        writer.add_scalar('Valid/pose_valid_cnt', pose_valid_meter.sum, epoch)
        if reproj_med_meter.count > 0:
            writer.add_scalar('Valid/reproj_err_median_px', reproj_med_meter.avg, epoch)
        writer.add_scalar('Valid/pnp_ransac_fail_cnt', ransac_fail_meter.sum, epoch)
        writer.add_scalar('Valid/pnp_ransac_ok_cnt', float(max(int(ransac_fail_meter.count) - float(ransac_fail_meter.sum), 0.0)), epoch)
        writer.add_scalar('Valid/pnp_ransac_fail_pct', (float(ransac_fail_meter.sum) / max(float(ransac_fail_meter.count), 1.0)) * 100.0, epoch)
        if ransac_inlier_meter.count > 0:
            writer.add_scalar('Valid/pnp_ransac_inlier_cnt', ransac_inlier_meter.avg, epoch)
            writer.add_scalar('Valid/pnp_ransac_inlier_cnt_median', _nanmedian(ransac_inlier_all), epoch)
        if err_t_ransac_meter.count > 0:
            writer.add_scalar('Valid/err_t_ransac [m]', err_t_ransac_meter.avg, epoch)
            writer.add_scalar('Valid/err_t_ransac_median [m]', _nanmedian(ransac_ok_err_t_all), epoch)
        if err_q_ransac_meter.count > 0:
            writer.add_scalar('Valid/err_q_ransac [deg]', err_q_ransac_meter.avg, epoch)
            writer.add_scalar('Valid/err_q_ransac_median [deg]', _nanmedian(ransac_ok_err_q_all), epoch)
        if err_t_meter.count > 0:
            writer.add_scalar('Valid/err_t_median [m]', _nanmedian(pnp_ok_err_t_all), epoch)
        if err_q_meter.count > 0:
            writer.add_scalar('Valid/err_q_median [deg]', _nanmedian(pnp_ok_err_q_all), epoch)

    # Aggregate different performances
    performances = {
        'eR': err_q_meter,
        'eT': err_t_meter,
        'speed (raw)': speed_meter,
        'speed (thr)': speed_meter_th
    }
    if has_keypts:
        performances['keypoint_rmse_px'] = rmse_meter
        performances['keypoint_rmse_px_median'] = _ScalarPerf(_nanmedian(keypoint_rmse_all), unit='pix')
        performances['keypoint_rmse_norm_input'] = rmse_norm_input_meter
        performances['keypoint_rmse_norm_input_median'] = _ScalarPerf(_nanmedian(keypoint_rmse_norm_input_all), unit='-')
        performances['keypoint_rmse_norm_bbox'] = rmse_norm_bbox_meter
        performances['keypoint_rmse_norm_bbox_median'] = _ScalarPerf(_nanmedian(keypoint_rmse_norm_bbox_all), unit='-')
        performances['pck_0.05_bbox_pct'] = pck05bbox_meter
        performances['pck_0.10_bbox_pct'] = pck10bbox_meter
        performances['keypoint_inside_pct'] = inside_meter
        performances['valid_kpt_pct'] = valid_kpt_meter
        if hm_peak_mean_meter.count > 0:
            performances['heatmap_peak_mean'] = hm_peak_mean_meter
            performances['heatmap_peak_std'] = hm_peak_std_meter
            performances['heatmap_entropy'] = hm_entropy_meter
            performances['heatmap_entropy_std'] = hm_entropy_std_meter
            performances['heatmap_entropy_beta'] = hm_entropy_beta_meter
            performances['heatmap_entropy_beta_std'] = hm_entropy_beta_std_meter
        if collapsed_kpt_dist_meter.count > 0:
            performances['collapsed_keypoint_distance'] = collapsed_kpt_dist_meter
            performances['collapsed_keypoint_spread_min_px'] = collapsed_kpt_spread_min_meter
        for k, m in enumerate(per_kpt_rmse_meters):
            if m.count > 0:
                performances[f'per_keypoint_rmse_px_{k:02d}'] = m
        for k, m in enumerate(per_kpt_pck5_meters):
            if m.count > 0:
                performances[f'per_keypoint_pck5_pct_{k:02d}'] = m
        for k, m in enumerate(per_kpt_pck10_meters):
            if m.count > 0:
                performances[f'per_keypoint_pck10_pct_{k:02d}'] = m
        for k, m in enumerate(per_kpt_pck05bbox_meters):
            if m.count > 0:
                performances[f'per_keypoint_pck05_bbox_pct_{k:02d}'] = m
        for k, m in enumerate(per_kpt_pck10bbox_meters):
            if m.count > 0:
                performances[f'per_keypoint_pck10_bbox_pct_{k:02d}'] = m
        for k, m in enumerate(per_kpt_peak_meters):
            if m.count > 0:
                performances[f'per_keypoint_peak_{k:02d}'] = m
        for k, m in enumerate(per_kpt_peak2_meters):
            if m.count > 0:
                performances[f'per_keypoint_peak2_{k:02d}'] = m
        for k, m in enumerate(per_kpt_peak_ratio_meters):
            if m.count > 0:
                performances[f'per_keypoint_peak_ratio_{k:02d}'] = m
        for k, m in enumerate(per_kpt_top1_to_gt_dist_hm_px_meters):
            if m.count > 0:
                performances[f'per_keypoint_top1_to_gt_dist_hm_px_{k:02d}'] = m
        for k, m in enumerate(per_kpt_top2_to_gt_dist_hm_px_meters):
            if m.count > 0:
                performances[f'per_keypoint_top2_to_gt_dist_hm_px_{k:02d}'] = m

    performances['pnp_fail_cnt'] = _ScalarPerf(pnp_fail_meter.sum, unit='cnt')
    pnp_eval_cnt = int(pnp_fail_meter.count)
    pnp_fail_cnt = float(pnp_fail_meter.sum)
    pnp_ok_cnt = float(max(pnp_eval_cnt - pnp_fail_cnt, 0.0))
    pnp_fail_pct = (pnp_fail_cnt / max(float(pnp_eval_cnt), 1.0)) * 100.0
    performances['pnp_eval_cnt'] = _ScalarPerf(pnp_eval_cnt, unit='cnt')
    performances['pnp_ok_cnt'] = _ScalarPerf(pnp_ok_cnt, unit='cnt')
    performances['raw_epnp_ok_cnt'] = _ScalarPerf(pnp_ok_cnt, unit='cnt')
    performances['pnp_fail_pct'] = _ScalarPerf(pnp_fail_pct, unit='%')
    performances['pose_valid_cnt'] = _ScalarPerf(pose_valid_meter.sum, unit='cnt')
    if reproj_med_meter.count > 0:
        performances['reproj_err_median_px'] = reproj_med_meter
        performances['reprojection_error_median_px'] = reproj_med_meter
    performances['eT_median'] = _ScalarPerf(_nanmedian(pnp_ok_err_t_all), unit='m')
    performances['eR_median'] = _ScalarPerf(_nanmedian(pnp_ok_err_q_all), unit='deg')

    performances['pnp_ransac_fail_cnt'] = _ScalarPerf(ransac_fail_meter.sum, unit='cnt')
    ransac_eval_cnt = int(ransac_fail_meter.count)
    ransac_fail_cnt = float(ransac_fail_meter.sum)
    ransac_ok_cnt = float(max(ransac_eval_cnt - ransac_fail_cnt, 0.0))
    ransac_fail_pct = (ransac_fail_cnt / max(float(ransac_eval_cnt), 1.0)) * 100.0
    performances['pnp_ransac_eval_cnt'] = _ScalarPerf(ransac_eval_cnt, unit='cnt')
    performances['pnp_ransac_ok_cnt'] = _ScalarPerf(ransac_ok_cnt, unit='cnt')
    performances['ransac_ok_cnt'] = _ScalarPerf(ransac_ok_cnt, unit='cnt')
    performances['pnp_ransac_fail_pct'] = _ScalarPerf(ransac_fail_pct, unit='%')
    if ransac_inlier_meter.count > 0:
        performances['pnp_ransac_inlier_cnt'] = ransac_inlier_meter
        performances['pnp_ransac_inlier_cnt_median'] = _ScalarPerf(_nanmedian(ransac_inlier_all), unit='cnt')
    if err_t_ransac_meter.count > 0:
        performances['eT_ransac'] = err_t_ransac_meter
        performances['eT_ransac_median'] = _ScalarPerf(_nanmedian(ransac_ok_err_t_all), unit='m')
    if err_q_ransac_meter.count > 0:
        performances['eR_ransac'] = err_q_ransac_meter
        performances['eR_ransac_median'] = _ScalarPerf(_nanmedian(ransac_ok_err_q_all), unit='deg')

    for k in range(nK):
        denom = int(ransac_inlier_cand_cnt[k])
        if denom <= 0:
            continue
        freq = float(ransac_inlier_sel_cnt[k]) / float(denom) * 100.0
        performances[f'per_keypoint_ransac_inlier_freq_pct_{k:02d}'] = _ScalarPerf(freq, unit='%')

    if (worst_k > 0 or best_k > 0) and has_keypts:
        import os
        import json
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(debug_dir, exist_ok=True)
        worst_sorted = sorted(
            worst,
            key=lambda d: (float(d['score']) if np.isfinite(d['score']) else float('inf')),
            reverse=True,
        )[:worst_k]
        best_sorted = sorted(
            best,
            key=lambda d: (float(d['score']) if np.isfinite(d['score']) else float('inf')),
        )[:best_k]

        H, W = int(cfg.input_shape[0]), int(cfg.input_shape[1])
        def _save_one(split, rank, item):
            sid = int(item['sample_id'])
            out_dir = osp.join(debug_dir, split, f'rank_{rank:02d}_idx_{sid:05d}')
            os.makedirs(out_dir, exist_ok=True)

            img = item['image'].mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
            plt.imsave(osp.join(out_dir, 'image.png'), img)

            sx = max(float(W - 1), 1.0)
            sy = max(float(H - 1), 1.0)
            kp_gt = item['kp_gt']
            kp_pr = item['kp_pr']
            vm = item['vm'].to(dtype=torch.bool)
            xs_gt = kp_gt[0].numpy() * sx
            ys_gt = kp_gt[1].numpy() * sy
            xs_pr = kp_pr[0].numpy() * sx
            ys_pr = kp_pr[1].numpy() * sy

            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.xlim(0, W - 1)
            plt.ylim(H - 1, 0)
            plt.scatter(xs_gt[vm.numpy()], ys_gt[vm.numpy()], c='lime', marker='+', label='gt_valid')
            plt.scatter(xs_pr[vm.numpy()], ys_pr[vm.numpy()], c='red', marker='x', label='pred_valid')
            plt.scatter(xs_gt[~vm.numpy()], ys_gt[~vm.numpy()], c='yellow', marker='.', s=10, label='gt_invalid')
            plt.axis('off')
            plt.legend(loc='lower right', fontsize=8)
            plt.savefig(osp.join(out_dir, 'kpts_overlay.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

            bb = item['bbox'].numpy()
            dx = max(float(bb[1] - bb[0]), 1.0)
            dy = max(float(bb[3] - bb[2]), 1.0)

            def _proj_to_crop(q, t):
                pts = project_keypoints(q, t, cameraMatrix, distCoeffs, corners3D.T)
                xs_p = (pts[0] - float(bb[0])) / dx * sx
                ys_p = (pts[1] - float(bb[2])) / dy * sy
                return xs_p, ys_p

            try:
                xs_p, ys_p = _proj_to_crop(item['q_pr'], item['t_pr'])
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.xlim(0, W - 1)
                plt.ylim(H - 1, 0)
                plt.scatter(xs_p, ys_p, c='cyan', marker='o', s=10, label='EPnP_proj')
                x0, x1 = float(np.nanmin(xs_p)), float(np.nanmax(xs_p))
                y0, y1 = float(np.nanmin(ys_p)), float(np.nanmax(ys_p))
                plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c='cyan', lw=1)
                plt.axis('off')
                plt.legend(loc='lower right', fontsize=8)
                plt.savefig(osp.join(out_dir, 'epnp_proj_wireframe.png'), bbox_inches='tight', pad_inches=0)
                plt.close()
            except Exception:
                pass

            try:
                xs_r, ys_r = _proj_to_crop(item['q_r'], item['t_r'])
                inl = np.asarray(item.get('inlier_idx', np.zeros((0,), dtype=np.int64))).reshape(-1)
                inl_mask = np.zeros((nK,), dtype=np.bool_)
                if inl.size > 0:
                    inl_mask[inl.clip(0, nK - 1)] = True
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.xlim(0, W - 1)
                plt.ylim(H - 1, 0)
                plt.scatter(xs_r[inl_mask], ys_r[inl_mask], c='lime', marker='o', s=12, label='RANSAC_inlier')
                plt.scatter(xs_r[~inl_mask], ys_r[~inl_mask], c='magenta', marker='x', s=18, label='RANSAC_outlier')
                plt.axis('off')
                plt.legend(loc='lower right', fontsize=8)
                plt.savefig(osp.join(out_dir, 'ransac_inlier_outlier.png'), bbox_inches='tight', pad_inches=0)
                plt.close()
            except Exception:
                pass

            pred_hm = item.get('pred_hm', None)
            if pred_hm is not None:
                hm = pred_hm.float().numpy()
                hm_max = np.max(hm, axis=0)
                hm_max = (hm_max - hm_max.min()) / max(float(hm_max.max() - hm_max.min()), 1e-8)
                hm_max_up = torch.nn.functional.interpolate(
                    torch.from_numpy(hm_max).to(dtype=torch.float32).view(1, 1, hm_max.shape[0], hm_max.shape[1]),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ).view(H, W).numpy()
                overlay = (
                    0.6 * img.astype(np.float32) / 255.0
                    + 0.4 * np.stack([hm_max_up, np.zeros_like(hm_max_up), np.zeros_like(hm_max_up)], axis=-1)
                )
                overlay = np.clip(overlay, 0.0, 1.0)
                plt.imsave(osp.join(out_dir, 'pred_heatmap_overlay.png'), overlay)

                gt_hm, _ = gaussian_heatmaps_from_keypoints(
                    kp_gt.unsqueeze(0),
                    pred_hm.shape[-2:],
                    sigma=float(getattr(cfg, 'heatmap_sigma', 2.0)),
                )
                gh = gt_hm.squeeze(0).detach().cpu().numpy()
                gh_max = np.max(gh, axis=0)
                gh_max = (gh_max - gh_max.min()) / max(float(gh_max.max() - gh_max.min()), 1e-8)
                gh_max_up = torch.nn.functional.interpolate(
                    torch.from_numpy(gh_max).to(dtype=torch.float32).view(1, 1, gh_max.shape[0], gh_max.shape[1]),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ).view(H, W).numpy()
                overlay2 = (
                    0.6 * img.astype(np.float32) / 255.0
                    + 0.4 * np.stack([np.zeros_like(gh_max_up), gh_max_up, np.zeros_like(gh_max_up)], axis=-1)
                )
                overlay2 = np.clip(overlay2, 0.0, 1.0)
                plt.imsave(osp.join(out_dir, 'gt_heatmap_overlay.png'), overlay2)

                coords3 = item.get('top3_coords', None)
                vals3 = item.get('top3_vals', None)
                pratio = item.get('peak_ratio', None)
                if coords3 is not None and vals3 is not None and pratio is not None:
                    np.savez_compressed(
                        osp.join(out_dir, 'peaks_top3.npz'),
                        top3_coords=coords3.numpy(),
                        top3_vals=vals3.numpy(),
                        peak_to_second_ratio=pratio.numpy(),
                    )

            per_err = item['per_err'].numpy()
            with open(osp.join(out_dir, 'per_keypoint_error_px.txt'), 'w', encoding='utf-8') as f:
                for k in range(min(per_err.shape[0], nK)):
                    f.write(f'k{k:02d}\t{float(per_err[k]):.6f}\tvalid={int(vm.numpy()[k])}\n')

            with open(osp.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'split': split,
                        'rank': int(rank),
                        'sample_id': int(sid),
                        'score_keypoint_rmse_px': float(item.get('score', float('inf'))),
                        'pnp_failed': bool(item.get('pnp_failed', False)),
                        'ransac_failed': bool(item.get('ransac_failed', False)),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        os.makedirs(osp.join(debug_dir, 'best'), exist_ok=True)
        os.makedirs(osp.join(debug_dir, 'worst'), exist_ok=True)

        for rank, item in enumerate(best_sorted):
            _save_one('best', rank, item)
        for rank, item in enumerate(worst_sorted):
            _save_one('worst', rank, item)

        with open(osp.join(debug_dir, 'best_list.json'), 'w', encoding='utf-8') as f:
            json.dump(
                [{'rank': int(i), 'sample_id': int(d['sample_id']), 'score': float(d['score'])} for i, d in enumerate(best_sorted)],
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(osp.join(debug_dir, 'worst_list.json'), 'w', encoding='utf-8') as f:
            json.dump(
                [{'rank': int(i), 'sample_id': int(d['sample_id']), 'score': float(d['score'])} for i, d in enumerate(worst_sorted)],
                f,
                ensure_ascii=False,
                indent=2,
            )

        hard = {
            'total_valid': hard_total_valid.tolist(),
            'invalid_cnt': hard_invalid.tolist(),
            'err_gt_10px_cnt': hard_err_gt10.tolist(),
            'err_gt_20px_cnt': hard_err_gt20.tolist(),
            'near_border_cnt': hard_near_border.tolist(),
            'near_border_err_gt_10px_cnt': hard_near_border_err_gt10.tolist(),
        }
        with open(osp.join(debug_dir, 'hard_keypoints.json'), 'w', encoding='utf-8') as f:
            json.dump(hard, f, indent=2, ensure_ascii=False)

        def _safe_mean(sum_arr, cnt_arr):
            out = []
            for i in range(int(sum_arr.shape[0])):
                c = float(cnt_arr[i])
                out.append(float(sum_arr[i] / c) if c > 0 else None)
            return out

        ctx = {
            'all_cnt': ctx_all_cnt.tolist(),
            'hard_cnt_err_gt_20px': ctx_hard_cnt.tolist(),
            'all_bbox_area_mean': _safe_mean(ctx_all_bbox_area_sum, ctx_all_cnt),
            'hard_bbox_area_mean': _safe_mean(ctx_hard_bbox_area_sum, ctx_hard_cnt),
            'all_bbox_min_side_mean': _safe_mean(ctx_all_bbox_min_sum, ctx_all_cnt),
            'hard_bbox_min_side_mean': _safe_mean(ctx_hard_bbox_min_sum, ctx_hard_cnt),
            'all_t_norm_mean': _safe_mean(ctx_all_tnorm_sum, ctx_all_cnt),
            'hard_t_norm_mean': _safe_mean(ctx_hard_tnorm_sum, ctx_hard_cnt),
            'all_rot_mag_deg_mean': _safe_mean(ctx_all_rotdeg_sum, ctx_all_cnt),
            'hard_rot_mag_deg_mean': _safe_mean(ctx_hard_rotdeg_sum, ctx_hard_cnt),
            'all_brightness_mean': _safe_mean(ctx_all_brightness_sum, ctx_all_cnt),
            'hard_brightness_mean': _safe_mean(ctx_hard_brightness_sum, ctx_hard_cnt),
            'all_border_dist_px_mean': _safe_mean(ctx_all_borderpx_sum, ctx_all_cnt),
            'hard_border_dist_px_mean': _safe_mean(ctx_hard_borderpx_sum, ctx_hard_cnt),
        }
        with open(osp.join(debug_dir, 'hard_keypoints_context.json'), 'w', encoding='utf-8') as f:
            json.dump(ctx, f, indent=2, ensure_ascii=False)

    # Write performances for each items
    with open(osp.join(cfg.logdir, 'err_q.txt'), 'w') as f:
        for eq in err_q_all:
            f.write('{:.5f}\n'.format(eq))

    with open(osp.join(cfg.logdir, 'err_t.txt'), 'w') as f:
        for et in err_t_all:
            f.write('{:.5f}\n'.format(et))

    with open(osp.join(cfg.logdir, 'speed_raw.txt'), 'w') as f:
        for spd in speed_raw_all:
            f.write('{:.5f}\n'.format(spd))

    with open(osp.join(cfg.logdir, 'speed_mod.txt'), 'w') as f:
        for spd in speed_mod_all:
            f.write('{:.5f}\n'.format(spd))

    try:
        pnp_eval_cnt = int(pnp_fail_meter.count)
        pnp_fail_cnt = float(pnp_fail_meter.sum)
        pnp_ok_cnt = float(max(pnp_eval_cnt - pnp_fail_cnt, 0.0))
        pnp_fail_pct = (pnp_fail_cnt / max(float(pnp_eval_cnt), 1.0)) * 100.0
        r_ok_cnt = float(max(int(ransac_fail_meter.count) - float(ransac_fail_meter.sum), 0.0))
        msg = (
            f'[small_valid] epoch={int(epoch)} '
            f'kp_rmse_mean={rmse_meter.avg:.3f} kp_rmse_med={_nanmedian(keypoint_rmse_all):.3f} '
            f'pnp_ok={int(pnp_ok_cnt)}/{int(pnp_eval_cnt)} pnp_fail_pct={pnp_fail_pct:.1f} '
            f'eT_mean={err_t_meter.avg:.4f} eT_med={_nanmedian(pnp_ok_err_t_all):.4f} '
            f'eR_mean={err_q_meter.avg:.3f} eR_med={_nanmedian(pnp_ok_err_q_all):.3f} '
            f'ransac_ok={int(r_ok_cnt)}/{int(ransac_fail_meter.count)} inlier_med={_nanmedian(ransac_inlier_all):.2f} '
            f'worst_dir={debug_dir}'
        )
        logger.info(msg)
    except Exception:
        pass

    return performances

def valid_spn(epoch, cfg, model, data_loader, cameraMatrix, distCoeffs, corners3D, writer, device, qClass):
    ''' Valid SPN model '''

    # Initialize trackers
    test_time_meter = AverageMeter('ms')
    err_q_meter     = AverageMeter('deg')
    err_t_meter     = AverageMeter('m')
    speed_meter     = AverageMeter('-')
    speed_meter_th  = AverageMeter('-')
    acc_meter       = AverageMeter('%')

    # switch to eval mode
    model.eval()

    # Loop through dataloader
    for idx, (images, bbox, q_gt, t_gt) in enumerate(data_loader):
        start = time.time()
        B     = images.shape[0]

        # Debug (uncomment)
        # imshow(images[0])
        # print(bbox[0])

        # Feed-forward
        with torch.no_grad():
            _, weights = model(images.to(device))

            # Post-processing (Orientation only)
            topWeights, topClasses = torch.topk(weights, cfg.num_neighbors, dim=1)
            topWeights = torch.softmax(topWeights, dim=1)

        for b in range(B):
            # Predicted quaternion classes
            qs_pr = qClass[topClasses[b].cpu()] # [N x 4]

            # Weighted mean
            q_pr = weighted_mean_quaternion(qs_pr, topWeights.cpu().squeeze())

            # Position
            t_pr = compute_position_spn(q_pr, bbox[b].numpy(), corners3D, cameraMatrix, distCoeffs)

            # Ground-truth
            q_gt_i = q_gt[b].numpy()
            t_gt_i = t_gt[b].numpy()

            # Metrics
            err_q = error_orientation(q_pr, q_gt_i) # [deg]
            err_t = error_translation(t_pr, t_gt_i)
            speed_raw, acc = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=False)
            speed_mod, _   = speed_score(t_pr, q_pr, t_gt_i, q_gt_i, applyThresh=True,
                   rotThresh=0.169, posThresh=0.002173)

        # Update
        test_time_meter.update((time.time()-start)*1000, B)
        err_q_meter.update(err_q, B)
        err_t_meter.update(err_t, B)
        speed_meter.update(speed_raw, B)
        speed_meter_th.update(speed_mod, B)
        acc_meter.update(acc*100, B)

        report_progress(epoch=epoch, lr=np.nan, epoch_iter=idx+1, epoch_size=len(data_loader),
                        time=test_time_meter, is_train=False, eT=err_t_meter, eR=err_q_meter,
                        speed=speed_meter, acc=acc_meter)

    # log to tensorboard
    if writer is not None:
        writer.add_scalar('Valid/err_q [deg]', err_q_meter.avg, epoch)
        writer.add_scalar('Valid/err_t [m]',   err_t_meter.avg, epoch)
        writer.add_scalar('Valid/speed (raw) [-]', speed_meter.avg, epoch)
        writer.add_scalar('Valid/speed (thr) [-]', speed_meter_th.avg, epoch)

    # Aggregate different performances
    performances = {
        'eR': err_q_meter,
        'eT': err_t_meter,
        'speed (raw)': speed_meter,
        'speed (thr)': speed_meter_th
    }

    return performances

def _keypts_to_pose(x_pr, y_pr, bbox, corners3D, cameraMatrix, distCoeffs=np.zeros((1,5))):
    ''' Convert detected keypoints to pose given RoI and camera properties
    Arguments:
        x_pr: (11,) torch.Tensor
        y_pr: (11,) torch.Tensor
        bbox: (4,)  torch.Tensor - Bounding box [xmin, xmax, ymin, ymax] (pix)
        ...
    Returns:
        q_pr: (4,) numpy.ndarray - Relative orientation as unit quaternion [qw, qx, qy, qz]
        t_pr: (3,) numpy.ndarray - Relative position in (m)
    '''
    if (not torch.isfinite(x_pr).all()) or (not torch.isfinite(y_pr).all()):
        raise ValueError('Non-finite keypoints')
    corners2D_pr = torch.cat((x_pr.unsqueeze(0), y_pr.unsqueeze(0)), dim=0) # [2 x 11]
    corners2D_pr = corners2D_pr.cpu().t().numpy() # [11 x 2]

    # Apply RoI
    xmin, xmax, ymin, ymax = bbox.numpy()
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * (xmax-xmin) + xmin
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * (ymax-ymin) + ymin
    spread = np.ptp(corners2D_pr, axis=0)
    if (not np.isfinite(spread).all()) or float(spread.min()) < 2.0:
        raise ValueError('Degenerate 2D keypoints for PnP')

    # Compute [R|t] by pnp
    q_pr, t_pr = pnp(corners3D, corners2D_pr, cameraMatrix, distCoeffs)
    if (not np.isfinite(q_pr).all()) or (not np.isfinite(t_pr).all()):
        raise ValueError('Non-finite pose from PnP')
    if float(np.linalg.norm(t_pr)) > 200.0:
        raise ValueError('Implausible translation from PnP')

    return q_pr, t_pr


def _keypts_to_pose_ransac(x_pr, y_pr, bbox, corners3D, cameraMatrix, distCoeffs=np.zeros((1,5)), reproj_thr_px=8.0):
    if (not torch.isfinite(x_pr).all()) or (not torch.isfinite(y_pr).all()):
        raise ValueError('Non-finite keypoints')
    corners2D_pr = torch.cat((x_pr.unsqueeze(0), y_pr.unsqueeze(0)), dim=0)
    corners2D_pr = corners2D_pr.cpu().t().numpy()

    xmin, xmax, ymin, ymax = bbox.numpy()
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * (xmax - xmin) + xmin
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * (ymax - ymin) + ymin
    spread = np.ptp(corners2D_pr, axis=0)
    if (not np.isfinite(spread).all()) or float(spread.min()) < 2.0:
        raise ValueError('Degenerate 2D keypoints for PnPRansac')

    q_pr, t_pr, inlier_idx = pnp_ransac(corners3D, corners2D_pr, cameraMatrix, distCoeffs, reprojectionError=float(reproj_thr_px))
    if (not np.isfinite(q_pr).all()) or (not np.isfinite(t_pr).all()):
        raise ValueError('Non-finite pose from PnPRansac')
    if float(np.linalg.norm(t_pr)) > 200.0:
        raise ValueError('Implausible translation from PnPRansac')

    return q_pr, t_pr, np.asarray(inlier_idx, dtype=np.int64).reshape(-1)

def _keypts_to_pose_masked(x_pr, y_pr, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask):
    m = valid_mask.detach().cpu().numpy().astype(np.bool_).reshape(-1)
    idx = np.where(m)[0]
    if idx.size < 4:
        raise ValueError('Insufficient valid keypoints for PnP')
    if (not torch.isfinite(x_pr).all()) or (not torch.isfinite(y_pr).all()):
        raise ValueError('Non-finite keypoints')

    corners2D_pr = torch.stack([x_pr, y_pr], dim=0).cpu().numpy().T  # (K,2)
    xmin, xmax, ymin, ymax = bbox.detach().cpu().numpy().reshape(4)
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * (xmax - xmin) + xmin
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * (ymax - ymin) + ymin

    p2 = corners2D_pr[idx, :]
    p3 = np.asarray(corners3D, dtype=np.float32).reshape(-1, 3)[idx, :]
    spread = np.ptp(p2, axis=0)
    if (not np.isfinite(spread).all()) or float(spread.min()) < 2.0:
        raise ValueError('Degenerate 2D keypoints for masked PnP')

    q_pr, t_pr = pnp(p3, p2, cameraMatrix, distCoeffs)
    if (not np.isfinite(q_pr).all()) or (not np.isfinite(t_pr).all()):
        raise ValueError('Non-finite pose from masked PnP')
    if float(np.linalg.norm(t_pr)) > 200.0:
        raise ValueError('Implausible translation from masked PnP')
    return q_pr, t_pr, idx

def _keypts_to_pose_ransac_masked(x_pr, y_pr, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask, reproj_thr_px=8.0):
    m = valid_mask.detach().cpu().numpy().astype(np.bool_).reshape(-1)
    idx = np.where(m)[0]
    if idx.size < 4:
        raise ValueError('Insufficient valid keypoints for PnPRansac')
    if (not torch.isfinite(x_pr).all()) or (not torch.isfinite(y_pr).all()):
        raise ValueError('Non-finite keypoints')

    corners2D_pr = torch.stack([x_pr, y_pr], dim=0).cpu().numpy().T  # (K,2)
    xmin, xmax, ymin, ymax = bbox.detach().cpu().numpy().reshape(4)
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * (xmax - xmin) + xmin
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * (ymax - ymin) + ymin

    p2 = corners2D_pr[idx, :]
    p3 = np.asarray(corners3D, dtype=np.float32).reshape(-1, 3)[idx, :]
    spread = np.ptp(p2, axis=0)
    if (not np.isfinite(spread).all()) or float(spread.min()) < 2.0:
        raise ValueError('Degenerate 2D keypoints for masked PnPRansac')

    q_pr, t_pr, inlier_local = pnp_ransac(p3, p2, cameraMatrix, distCoeffs, reprojectionError=float(reproj_thr_px))
    if (not np.isfinite(q_pr).all()) or (not np.isfinite(t_pr).all()):
        raise ValueError('Non-finite pose from masked PnPRansac')
    if float(np.linalg.norm(t_pr)) > 200.0:
        raise ValueError('Implausible translation from masked PnPRansac')
    inlier_local = np.asarray(inlier_local, dtype=np.int64).reshape(-1)
    inlier_global = idx[inlier_local] if inlier_local.size > 0 else np.zeros((0,), dtype=np.int64)
    return q_pr, t_pr, inlier_global, idx
