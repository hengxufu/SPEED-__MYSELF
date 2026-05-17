from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

import torch.nn.functional as F

from src.utils.heatmaps import gaussian_heatmaps_from_keypoints
from src.utils.heatmaps import heatmap_to_keypoints_argmax, heatmap_to_keypoints_softargmax
from src.utils.utils import pnp, pnp_ransac, project_keypoints, quat2dcm


def build_gt_heatmaps(keypoints_norm, heatmap_size, sigma):
    return gaussian_heatmaps_from_keypoints(keypoints_norm, heatmap_size, sigma=float(sigma))


def heatmap_to_keypoints(pred_hm, decode='argmax', beta=100.0):
    if str(decode).lower() == 'softargmax':
        return heatmap_to_keypoints_softargmax(pred_hm, beta=float(beta))
    return heatmap_to_keypoints_argmax(pred_hm)


def heatmap_loss(
    logits,
    gt_hm,
    valid_mask,
    loss_type='bce',
    pos_thr=0.1,
    neg_weight=0.01,
    coord_aux_weight=0.1,
    channel_weights=None,
):
    if str(loss_type).lower() in ('heatmap_ce_coord_aux', 'ce_coord_aux'):
        eps = 1e-8
        B, K, H, W = logits.shape
        flat_logits = logits.view(B, K, H * W)
        log_p = F.log_softmax(flat_logits, dim=-1)

        gt = torch.nan_to_num(gt_hm, nan=0.0, posinf=0.0, neginf=0.0)
        gt_flat = gt.view(B, K, H * W)
        gt_sum = gt_flat.sum(dim=-1, keepdim=True)
        q = gt_flat / (gt_sum.clamp(min=eps))
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        valid_c = valid_mask.to(dtype=torch.bool)
        m = valid_c.to(dtype=log_p.dtype)

        kl = F.kl_div(log_p, q, reduction='none')
        kl = kl.sum(dim=-1)

        p = torch.softmax(flat_logits, dim=-1)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=logits.device, dtype=torch.float32),
            torch.arange(W, device=logits.device, dtype=torch.float32),
            indexing='ij',
        )
        grid_x = xx.reshape(1, 1, H * W)
        grid_y = yy.reshape(1, 1, H * W)
        ex = (p * grid_x).sum(dim=-1) / max(float(W - 1), 1.0)
        ey = (p * grid_y).sum(dim=-1) / max(float(H - 1), 1.0)

        gt_kp, _ = heatmap_to_keypoints_argmax(gt_hm)
        gx = gt_kp[:, 0, :]
        gy = gt_kp[:, 1, :]
        coord_pred = torch.stack([ex, ey], dim=-1)
        coord_gt = torch.stack([gx, gy], dim=-1)
        coord_l1 = F.smooth_l1_loss(coord_pred, coord_gt, reduction='none').sum(dim=-1)

        cw = channel_weights
        if cw is not None:
            cw = torch.as_tensor(cw, dtype=kl.dtype, device=kl.device).view(1, -1)
        else:
            cw = torch.ones((1, K), dtype=kl.dtype, device=kl.device)

        denom = torch.clamp((m * cw).sum(dim=1), min=1.0)
        dist_loss = ((kl * m) * cw).sum(dim=1) / denom
        coord_loss = ((coord_l1 * m) * cw).sum(dim=1) / denom

        total = dist_loss + float(coord_aux_weight) * coord_loss
        return total.mean(), dist_loss.mean(), coord_loss.mean()

    valid_c = valid_mask.to(dtype=torch.bool)
    valid_hw = valid_c.unsqueeze(-1).unsqueeze(-1)
    pos_hw = (gt_hm > float(pos_thr)) & valid_hw
    neg_hw = (~pos_hw) & valid_hw

    if str(loss_type).lower() == 'mse':
        diff = torch.nn.functional.mse_loss(logits, gt_hm, reduction='none')
    else:
        diff = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt_hm, reduction='none')

    pos_cnt = torch.clamp(pos_hw.to(dtype=diff.dtype).sum(dim=(-2, -1)), min=1.0)
    neg_cnt = torch.clamp(neg_hw.to(dtype=diff.dtype).sum(dim=(-2, -1)), min=1.0)
    pos_loss = (diff * pos_hw.to(dtype=diff.dtype)).sum(dim=(-2, -1)) / pos_cnt
    neg_loss = (diff * neg_hw.to(dtype=diff.dtype)).sum(dim=(-2, -1)) / neg_cnt

    per_k_loss = pos_loss + float(neg_weight) * neg_loss
    vc = valid_c.to(dtype=per_k_loss.dtype)
    denom_c = torch.clamp(vc.sum(dim=1), min=1.0)
    loss = (per_k_loss * vc).sum(dim=1) / denom_c
    loss = loss.mean()

    pos_mean = (pos_loss * vc).sum(dim=1) / denom_c
    neg_mean = (neg_loss * vc).sum(dim=1) / denom_c
    pos_mean = pos_mean.mean()
    neg_mean = neg_mean.mean()
    return loss, pos_mean, neg_mean


def geom_valid_mask(q_gt, t_gt, bbox, corners3D, cameraMatrix, distCoeffs, heatmap_size, z_min=1e-6):
    q = np.array(q_gt, dtype=np.float64).reshape(4)
    t = np.array(t_gt, dtype=np.float64).reshape(3)
    bb = np.array(bbox, dtype=np.float64).reshape(-1)
    if bb.size < 4 or (not np.isfinite(bb[:4]).all()):
        return np.zeros((int(corners3D.shape[0]),), dtype=bool)
    bb0, bb1, bb2, bb3 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    dx = max(float(bb1 - bb0), 1.0)
    dy = max(float(bb3 - bb2), 1.0)

    pts = np.array(corners3D, dtype=np.float64).reshape(-1, 3)
    keypoints = np.transpose(pts)
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]), dtype=np.float64)))
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(t, 1)))
    xyz = np.dot(pose_mat, keypoints)
    zc = xyz[2, :]
    z_ok = np.isfinite(zc) & (zc > float(z_min))

    pts2d = project_keypoints(q, t, cameraMatrix, distCoeffs, pts)
    u = np.array(pts2d[0], dtype=np.float64).reshape(-1)
    v = np.array(pts2d[1], dtype=np.float64).reshape(-1)
    uv_ok = np.isfinite(u) & np.isfinite(v)
    in_crop = (u >= bb0) & (u <= bb1) & (v >= bb2) & (v <= bb3)

    hm_h = int(heatmap_size[0])
    hm_w = int(heatmap_size[1])
    u_hm = (u - bb0) / dx * max(float(hm_w - 1), 1.0)
    v_hm = (v - bb2) / dy * max(float(hm_h - 1), 1.0)
    in_hm = (u_hm >= 0.0) & (u_hm <= float(hm_w - 1)) & (v_hm >= 0.0) & (v_hm <= float(hm_h - 1))

    return z_ok & uv_ok & in_crop & in_hm


def keypoints_to_pose(kp_norm, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask=None):
    corners2D = torch.stack([kp_norm[0], kp_norm[1]], dim=0).t().cpu().numpy()
    bb = np.array(bbox, dtype=np.float64).reshape(4)
    corners2D[:, 0] = corners2D[:, 0] * (bb[1] - bb[0]) + bb[0]
    corners2D[:, 1] = corners2D[:, 1] * (bb[3] - bb[2]) + bb[2]
    pts3d = np.array(corners3D, dtype=np.float32).reshape(-1, 3)

    idx = np.arange(pts3d.shape[0], dtype=np.int64)
    if valid_mask is not None:
        m = np.array(valid_mask, dtype=bool).reshape(-1)
        idx = np.where(m)[0]
        if int(idx.size) < 4:
            raise ValueError('Insufficient valid keypoints for PnP')
        corners2D = corners2D[idx]
        pts3d = pts3d[idx]

    spread = np.ptp(corners2D, axis=0)
    if (not np.isfinite(spread).all()) or float(spread.min()) < 2.0:
        raise ValueError('Degenerate 2D keypoints for PnP')

    q_pr, t_pr = pnp(pts3d, corners2D, cameraMatrix, distCoeffs)
    if (not np.isfinite(q_pr).all()) or (not np.isfinite(t_pr).all()):
        raise ValueError('Non-finite pose from PnP')
    if float(np.linalg.norm(t_pr)) > 200.0:
        raise ValueError('Implausible translation from PnP')
    return q_pr, t_pr, idx


def keypoints_to_pose_ransac(kp_norm, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask=None, reproj_thr_px=8.0):
    corners2D = torch.stack([kp_norm[0], kp_norm[1]], dim=0).t().cpu().numpy()
    bb = np.array(bbox, dtype=np.float64).reshape(4)
    corners2D[:, 0] = corners2D[:, 0] * (bb[1] - bb[0]) + bb[0]
    corners2D[:, 1] = corners2D[:, 1] * (bb[3] - bb[2]) + bb[2]
    pts3d = np.array(corners3D, dtype=np.float32).reshape(-1, 3)

    idx = np.arange(pts3d.shape[0], dtype=np.int64)
    if valid_mask is not None:
        m = np.array(valid_mask, dtype=bool).reshape(-1)
        idx = np.where(m)[0]
        if int(idx.size) < 4:
            raise ValueError('Insufficient valid keypoints for PnPRansac')
        corners2D = corners2D[idx]
        pts3d = pts3d[idx]

    spread = np.ptp(corners2D, axis=0)
    if (not np.isfinite(spread).all()) or float(spread.min()) < 2.0:
        raise ValueError('Degenerate 2D keypoints for PnPRansac')

    q_pr, t_pr, inlier_idx = pnp_ransac(pts3d, corners2D, cameraMatrix, distCoeffs, reprojectionError=float(reproj_thr_px))
    if (not np.isfinite(q_pr).all()) or (not np.isfinite(t_pr).all()):
        raise ValueError('Non-finite pose from PnPRansac')
    if float(np.linalg.norm(t_pr)) > 200.0:
        raise ValueError('Implausible translation from PnPRansac')

    inlier_idx = np.array(inlier_idx, dtype=np.int64).reshape(-1)
    inlier_global = idx[inlier_idx] if inlier_idx.size > 0 else np.zeros((0,), dtype=np.int64)
    return q_pr, t_pr, inlier_global, idx


def reprojection_errors_px(kp_norm, bbox, q_pr, t_pr, corners3D, cameraMatrix, distCoeffs):
    bb = np.array(bbox, dtype=np.float64).reshape(4)
    dx = max(float(bb[1] - bb[0]), 1.0)
    dy = max(float(bb[3] - bb[2]), 1.0)

    pred = torch.stack([kp_norm[0], kp_norm[1]], dim=0).t().cpu().numpy()
    pred[:, 0] = pred[:, 0] * dx + bb[0]
    pred[:, 1] = pred[:, 1] * dy + bb[2]

    proj = project_keypoints(np.array(q_pr, dtype=np.float64), np.array(t_pr, dtype=np.float64), cameraMatrix, distCoeffs, np.array(corners3D, dtype=np.float64))
    proj = np.stack([proj[0], proj[1]], axis=1)

    err = np.sqrt(np.sum((proj - pred) ** 2, axis=1))
    return err


def pose_valid(q_pr, t_pr, reproj_err_px, valid_mask=None, min_valid_k=4, reproj_median_thr_px=10.0, t_norm_range=(0.5, 20.0)):
    if q_pr is None or t_pr is None:
        return False
    q = np.array(q_pr, dtype=np.float64).reshape(4)
    t = np.array(t_pr, dtype=np.float64).reshape(3)
    if (not np.isfinite(q).all()) or (not np.isfinite(t).all()):
        return False
    qn = float(np.linalg.norm(q))
    if (not np.isfinite(qn)) or qn <= 1e-6:
        return False
    tn = float(np.linalg.norm(t))
    if (not np.isfinite(tn)) or (tn < float(t_norm_range[0])) or (tn > float(t_norm_range[1])):
        return False

    if valid_mask is not None:
        m = np.array(valid_mask, dtype=bool).reshape(-1)
        if int(m.sum()) < int(min_valid_k):
            return False
        err = np.array(reproj_err_px, dtype=np.float64).reshape(-1)[m]
    else:
        err = np.array(reproj_err_px, dtype=np.float64).reshape(-1)
    if err.size == 0:
        return False
    med = float(np.median(err[np.isfinite(err)])) if np.isfinite(err).any() else float('inf')
    return med < float(reproj_median_thr_px)
