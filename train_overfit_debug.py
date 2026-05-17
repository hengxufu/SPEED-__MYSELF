import os
import os.path as osp
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.datasets.transforms import build_transforms
from src.nets.build import get_model
from src.utils.utils import set_all_seeds, load_tango_3d_keypoints, load_camera_intrinsics, pnp, pnp_ransac, project_keypoints, quat2dcm
from src.utils.metrics import error_orientation, error_translation, speed_score
from src.utils.heatmaps import keypoint_rmse_pixels, inside_image_percentage
from src.utils.heatmaps import per_keypoint_error_pixels, heatmap_peaks, heatmap_peak_to_second_ratio
from src.utils.heatmap_pipeline import build_gt_heatmaps, heatmap_to_keypoints, heatmap_loss, geom_valid_mask, keypoints_to_pose, keypoints_to_pose_ransac


def _resolve_root_and_csv(dataroot, dataname, domain, csv_rel):
    dr = osp.abspath(dataroot)
    if osp.isabs(csv_rel):
        if osp.exists(csv_rel):
            base = osp.join(dr, dataname) if dataname else dr
            base = base if osp.exists(base) else dr
            return base, csv_rel
        raise FileNotFoundError(f'CSV not found: {csv_rel}')

    candidates = []
    if dataname:
        candidates.append(osp.join(dr, dataname))
    candidates.append(dr)

    tried = []
    for root in candidates:
        csv_path = osp.join(root, domain, csv_rel)
        tried.append(csv_path)
        if osp.exists(csv_path):
            return root, csv_path
    raise FileNotFoundError('CSV not found. Tried:\n' + '\n'.join(tried))


class OverfitDataset(Dataset):
    def __init__(self, root, csv_path, num_keypoints=11, transforms=None, limit=32):
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


def _pnp_try(kp_norm, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask=None):
    q_pr, t_pr, _ = keypoints_to_pose(kp_norm, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask=valid_mask)
    return q_pr, t_pr


def _pnp_ransac_try(kp_norm, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask=None, reproj_thr_px=8.0):
    q_pr, t_pr, inlier_idx, _ = keypoints_to_pose_ransac(
        kp_norm, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask=valid_mask, reproj_thr_px=reproj_thr_px)
    return q_pr, t_pr, int(np.asarray(inlier_idx).reshape(-1).shape[0])


def _camera_frame_xyz(q_vbs2tango, r_Vo2To_vbs, points_3d):
    q = np.array(q_vbs2tango, dtype=np.float64).reshape(4)
    t = np.array(r_Vo2To_vbs, dtype=np.float64).reshape(3)
    pts = np.array(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3)
    R = quat2dcm(q)
    xyz = (R.T @ pts.T).T + t.reshape(1, 3)
    return xyz


def _foreground_valid_mask(image, kp_norm, fg_thr=0.12, patch=3):
    img = image.detach()
    if img.ndim != 3:
        raise ValueError('image must be CHW')
    C, H, W = img.shape
    gray = img.float().mean(dim=0)
    K = int(kp_norm.shape[-1])
    xs = kp_norm[0].detach().cpu().numpy().reshape(-1)
    ys = kp_norm[1].detach().cpu().numpy().reshape(-1)
    out = np.zeros((K,), dtype=bool)
    if patch < 1:
        patch = 1
    r = int(patch // 2)
    for k in range(K):
        x = xs[k]
        y = ys[k]
        if (not np.isfinite(x)) or (not np.isfinite(y)):
            continue
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            continue
        ix = int(round(float(x) * max(float(W - 1), 1.0)))
        iy = int(round(float(y) * max(float(H - 1), 1.0)))
        x0 = max(ix - r, 0)
        x1 = min(ix + r + 1, W)
        y0 = max(iy - r, 0)
        y1 = min(iy + r + 1, H)
        v = float(gray[y0:y1, x0:x1].mean().detach().cpu())
        out[k] = (v > float(fg_thr))
    return out


def _eval_overfit(model, dl_eval, device, cfg, corners3D, cameraMatrix, distCoeffs, it, savedir, save_debug=False):
    model.eval()
    rmses = []
    worst = []
    peaks_mean = []
    peaks_per_k_sum = None
    peaks_per_k_cnt = None
    peak_ratio_per_k_sum = None
    peak_ratio_per_k_cnt = None
    per_k_err_sum = None
    per_k_err_cnt = None
    pck5_sum = None
    pck10_sum = None
    pck_cnt = None
    pos_loss_sum = None
    neg_loss_sum = None
    loss_cnt = None

    pnp_eval = 0
    pnp_ok = 0
    eT_ok = []
    eR_ok = []
    sp_raw_ok = []
    sp_thr_ok = []

    ransac_eval = 0
    ransac_ok = 0
    ransac_inliers_ok = []
    eT_r_ok = []
    eR_r_ok = []

    overfit_eval_count = 0

    with torch.no_grad():
        for images, bbox, keypts, q_gt, t_gt in dl_eval:
            images = images.to(device)
            keypts = keypts.to(device)
            loss, sm = model(images, keypts)

            pred_hm = getattr(model, 'last_pred_heatmaps', None)
            pred_logits = getattr(model, 'last_pred_heatmaps_logits', None)
            if pred_hm is None:
                xc, yc = model(images)
                pred_kp = torch.stack([xc.to(device), yc.to(device)], dim=1)
                pred_hm = None
                pred_logits = None
            else:
                pred_kp, _ = heatmap_to_keypoints(pred_hm, decode=getattr(cfg, 'heatmap_decode', 'argmax'), beta=float(getattr(cfg, 'heatmap_beta', 100.0)))

            valid = torch.isfinite(keypts[:, 0, :]) & torch.isfinite(keypts[:, 1, :]) & (keypts[:, 0, :] >= 0) & (keypts[:, 0, :] <= 1) & (keypts[:, 1, :] >= 0) & (keypts[:, 1, :] <= 1)

            B = images.shape[0]
            for b in range(B):
                overfit_eval_count += 1
                vm = valid[b]
                if bool(getattr(cfg, 'geom_valid_mask', False)):
                    hm_size = None
                    if pred_hm is not None:
                        hm_size = (int(pred_hm.shape[-2]), int(pred_hm.shape[-1]))
                    elif pred_logits is not None:
                        hm_size = (int(pred_logits.shape[-2]), int(pred_logits.shape[-1]))
                    else:
                        hm_size = (int(getattr(cfg, 'heatmap_size', (56, 56))[0]), int(getattr(cfg, 'heatmap_size', (56, 56))[1]))
                    gm = geom_valid_mask(
                        q_gt[b].detach().cpu().numpy(),
                        t_gt[b].detach().cpu().numpy(),
                        bbox[b].detach().cpu().numpy(),
                        corners3D,
                        cameraMatrix,
                        distCoeffs,
                        heatmap_size=hm_size,
                        z_min=float(getattr(cfg, 'z_min', 1e-6)),
                    )
                    gm_t = torch.as_tensor(gm, dtype=torch.bool, device=vm.device)
                    vm = vm & gm_t

                rmse_b = keypoint_rmse_pixels(pred_kp[b:b+1], keypts[b:b+1], image_size=cfg.input_shape, valid_mask=vm.unsqueeze(0))
                rmse_val = float(rmse_b.detach().cpu())
                rmses.append(rmse_val)
                worst.append((rmse_val, int(overfit_eval_count - 1)))

                per_err = per_keypoint_error_pixels(pred_kp[b:b+1], keypts[b:b+1], image_size=cfg.input_shape).squeeze(0).detach()
                m = vm.to(dtype=per_err.dtype)
                if per_k_err_sum is None:
                    per_k_err_sum = torch.zeros_like(per_err)
                    per_k_err_cnt = torch.zeros_like(per_err)
                per_k_err_sum += per_err * m
                per_k_err_cnt += m

                ok5 = ((per_err <= 5.0).to(dtype=per_err.dtype) * m)
                ok10 = ((per_err <= 10.0).to(dtype=per_err.dtype) * m)
                if pck5_sum is None:
                    pck5_sum = torch.zeros_like(per_err)
                    pck10_sum = torch.zeros_like(per_err)
                    pck_cnt = torch.zeros_like(per_err)
                pck5_sum += ok5
                pck10_sum += ok10
                pck_cnt += m

                if pred_hm is not None:
                    pk = heatmap_peaks(pred_hm[b:b+1]).squeeze(0).detach()
                    m2 = vm.to(dtype=pk.dtype)
                    denom = float(torch.clamp(m2.sum(), min=1.0).detach().cpu())
                    peaks_mean.append(float((pk * m2).sum().detach().cpu() / denom))
                    if peaks_per_k_sum is None:
                        peaks_per_k_sum = torch.zeros_like(pk)
                        peaks_per_k_cnt = torch.zeros_like(pk)
                    peaks_per_k_sum += pk * m2
                    peaks_per_k_cnt += m2

                    pr = heatmap_peak_to_second_ratio(pred_hm[b:b+1]).squeeze(0).detach()
                    if peak_ratio_per_k_sum is None:
                        peak_ratio_per_k_sum = torch.zeros_like(pr)
                        peak_ratio_per_k_cnt = torch.zeros_like(pr)
                    peak_ratio_per_k_sum += pr * m2
                    peak_ratio_per_k_cnt += m2

                if pred_logits is not None:
                    gt_hm, valid_hm = build_gt_heatmaps(keypts[b:b+1], pred_logits.shape[-2:], float(getattr(cfg, 'heatmap_sigma', 2.0)))
                    gt_hm = gt_hm.squeeze(0)
                    valid_hm = valid_hm.squeeze(0)
                    vmk = valid_hm.to(dtype=torch.bool) & vm.to(dtype=torch.bool)
                    pos_hw = (gt_hm > float(getattr(cfg, 'heatmap_pos_thr', 0.1))) & vmk.unsqueeze(-1).unsqueeze(-1)
                    neg_hw = (~pos_hw) & vmk.unsqueeze(-1).unsqueeze(-1)
                    _, pos_l, neg_l = heatmap_loss(
                        pred_logits[b:b+1],
                        gt_hm.unsqueeze(0),
                        vmk.unsqueeze(0),
                        loss_type=getattr(cfg, 'heatmap_loss', 'bce'),
                        pos_thr=float(getattr(cfg, 'heatmap_pos_thr', 0.1)),
                        neg_weight=float(getattr(cfg, 'heatmap_neg_weight', 0.01)),
                    )
                    vc = vmk.to(dtype=pos_l.dtype)
                    denom_c = torch.clamp(vc.sum(), min=1.0)
                    if pos_loss_sum is None:
                        pos_loss_sum = 0.0
                        neg_loss_sum = 0.0
                        loss_cnt = 0.0
                    pos_loss_sum += float((pos_l * vc).sum().detach().cpu() / denom_c)
                    neg_loss_sum += float((neg_l * vc).sum().detach().cpu() / denom_c)
                    loss_cnt += 1.0

                pnp_eval += 1
                try:
                    q_pr, t_pr = _pnp_try(pred_kp[b].detach().cpu(), bbox[b], corners3D, cameraMatrix, distCoeffs, valid_mask=vm)
                    pnp_ok += 1
                    qg = q_gt[b].detach().cpu().numpy()
                    tg = t_gt[b].detach().cpu().numpy()
                    eR = error_orientation(q_pr, qg)
                    eT = error_translation(t_pr, tg)
                    sp_raw, _ = speed_score(t_pr, q_pr, tg, qg, applyThresh=False)
                    sp_thr, _ = speed_score(t_pr, q_pr, tg, qg, applyThresh=True, rotThresh=0.169, posThresh=0.002173)
                    eT_ok.append(float(eT))
                    eR_ok.append(float(eR))
                    sp_raw_ok.append(float(sp_raw))
                    sp_thr_ok.append(float(sp_thr))
                except Exception:
                    pass

                ransac_eval += 1
                try:
                    qg = q_gt[b].detach().cpu().numpy()
                    tg = t_gt[b].detach().cpu().numpy()
                    q_r, t_r, inl = _pnp_ransac_try(pred_kp[b].detach().cpu(), bbox[b], corners3D, cameraMatrix, distCoeffs, valid_mask=vm, reproj_thr_px=8.0)
                    ransac_ok += 1
                    ransac_inliers_ok.append(int(inl))
                    eRr = error_orientation(q_r, qg)
                    eTr = error_translation(t_r, tg)
                    eT_r_ok.append(float(eTr))
                    eR_r_ok.append(float(eRr))
                except Exception:
                    pass

                if save_debug:
                    os.makedirs(savedir, exist_ok=True)
                    pk_mean = float('nan')
                    bb0 = bb1 = bb2 = bb3 = float('nan')
                    img_w = img_h = -1
                    hm_w = hm_h = -1
                    scx = scy = float('nan')
                    if pred_hm is not None:
                        pk = heatmap_peaks(pred_hm[b:b+1]).squeeze(0).detach()
                        m2 = vm.to(dtype=pk.dtype)
                        pk_mean = float((pk * m2).sum().detach().cpu() / max(float(torch.clamp(m2.sum(), min=1.0).detach().cpu()), 1.0))
                        hm_h = int(pred_hm.shape[-2])
                        hm_w = int(pred_hm.shape[-1])
                    kp_gt_n = keypts[b].detach().cpu()
                    kp_pr_n = pred_kp[b].detach().cpu()
                    bb = bbox[b].numpy().reshape(-1)
                    if bb.size >= 4 and np.isfinite(bb[:4]).all():
                        bb0, bb1, bb2, bb3 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                        dx = max(float(bb1 - bb0), 1.0)
                        dy = max(float(bb3 - bb2), 1.0)
                        img_h = int(images[b].shape[-2])
                        img_w = int(images[b].shape[-1])
                        sx = max(float(img_w - 1), 1.0)
                        sy = max(float(img_h - 1), 1.0)
                        scx = sx / dx
                        scy = sy / dy
                    tag = (
                        f'iter{it:05d}_idx{overfit_eval_count-1:03d}_rmse{rmses[-1]:.3f}_peak{pk_mean:.3f}'
                        f'_img{img_w}x{img_h}_hm{hm_w}x{hm_h}'
                        f'_bb{bb0:.1f}-{bb1:.1f}-{bb2:.1f}-{bb3:.1f}'
                        f'_sc{scx:.6f}-{scy:.6f}'
                    )
                    try:
                        q_pr, t_pr = _pnp_try(kp_pr_n, bbox[b], corners3D, cameraMatrix, distCoeffs, valid_mask=vm)
                        pts2d_proj = project_keypoints(q_pr, t_pr, cameraMatrix, distCoeffs, corners3D.T)
                        _, h, w = images[b].shape
                        sx = max(float(w - 1), 1.0)
                        sy = max(float(h - 1), 1.0)
                        dx = max(float(bb[1] - bb[0]), 1.0)
                        dy = max(float(bb[3] - bb[2]), 1.0)
                        xs_p = (pts2d_proj[0] - float(bb[0])) / dx * sx
                        ys_p = (pts2d_proj[1] - float(bb[2])) / dy * sy
                        _save_debug_image(
                            osp.join(savedir, tag + '_kpt.png'),
                            images[b].detach().cpu(),
                            kp_gt_n,
                            kp_pr_n,
                            kp_pnp=(xs_p, ys_p),
                        )
                    except Exception:
                        _save_debug_image(
                            osp.join(savedir, tag + '_kpt_pnp_fail.png'),
                            images[b].detach().cpu(),
                            kp_gt_n,
                            kp_pr_n,
                        )
                    if pred_hm is not None:
                        gt_hm_vis = None
                        if pred_logits is not None:
                            gt_hm_vis, _ = build_gt_heatmaps(kp_gt_n.unsqueeze(0), pred_hm.shape[-2:], float(getattr(cfg, 'heatmap_sigma', 2.0)))
                            gt_hm_vis = gt_hm_vis.squeeze(0)
                            _save_joint_overlay(
                                osp.join(savedir, tag + '_joint.png'),
                                images[b].detach().cpu(),
                                kp_gt=kp_gt_n,
                                kp_pr=kp_pr_n,
                                pred_hm=pred_hm[b].detach().cpu(),
                                gt_hm=gt_hm_vis.detach().cpu(),
                                bbox=bb[:4],
                            )
                        _save_pred_heatmap_overlay(
                            osp.join(savedir, tag + '_predhm.png'),
                            images[b].detach().cpu(),
                            pred_hm[b].detach().cpu(),
                            kp_gt=kp_gt_n,
                        )
                        if pred_logits is not None:
                            gt_hm, _ = build_gt_heatmaps(kp_gt_n.unsqueeze(0), pred_hm.shape[-2:], float(getattr(cfg, 'heatmap_sigma', 2.0)))
                            gt_hm = gt_hm.squeeze(0)
                            _save_pred_heatmap_overlay(
                                osp.join(savedir, tag + '_gthm.png'),
                                images[b].detach().cpu(),
                                gt_hm.detach().cpu(),
                                kp_gt=kp_gt_n,
                            )
                            ch_dir = osp.join(savedir, tag + '_channels')
                            _save_heatmap_channels(ch_dir, pred_hm[b], prefix='pred', vmax=1.0)
                            _save_heatmap_channels(ch_dir, gt_hm, prefix='gt', vmax=1.0)

    rmse_mean = float(np.mean(rmses)) if len(rmses) else float('nan')
    rmse_median = float(np.median(rmses)) if len(rmses) else float('nan')
    peak_mean = float(np.mean(peaks_mean)) if len(peaks_mean) else float('nan')
    worst = sorted(worst, key=lambda x: -x[0])[:5]

    per_k_rmse = None
    if per_k_err_sum is not None:
        denom = torch.clamp(per_k_err_cnt, min=1.0)
        per_k_rmse = (per_k_err_sum / denom).cpu().numpy().tolist()

    peak_per_k = None
    if peaks_per_k_sum is not None:
        denom = torch.clamp(peaks_per_k_cnt, min=1.0)
        peak_per_k = (peaks_per_k_sum / denom).cpu().numpy().tolist()

    peak_ratio_per_k = None
    if peak_ratio_per_k_sum is not None:
        denom = torch.clamp(peak_ratio_per_k_cnt, min=1.0)
        peak_ratio_per_k = (peak_ratio_per_k_sum / denom).cpu().numpy().tolist()

    pck5_per_k = None
    pck10_per_k = None
    if pck5_sum is not None:
        denom = torch.clamp(pck_cnt, min=1.0)
        pck5_per_k = (pck5_sum / denom).cpu().numpy().tolist()
        pck10_per_k = (pck10_sum / denom).cpu().numpy().tolist()

    per_k_loss = None

    pnp_fail = int(max(pnp_eval - pnp_ok, 0))
    pnp_fail_pct = (float(pnp_fail) / max(float(pnp_eval), 1.0)) * 100.0

    out = {
        'overfit_eval_count': int(overfit_eval_count),
        'rmse_px_mean': float(rmse_mean),
        'rmse_px_median': float(rmse_median),
        'rmse_worst5': worst,
        'peak_mean': float(peak_mean),
        'per_keypoint_rmse_px': per_k_rmse,
        'heatmap_peak_per_keypoint': peak_per_k,
        'heatmap_peak_ratio_per_keypoint': peak_ratio_per_k,
        'per_keypoint_pck_5px': pck5_per_k,
        'per_keypoint_pck_10px': pck10_per_k,
        'per_keypoint_loss': per_k_loss,
        'pos_loss_mean': float(pos_loss_sum / max(loss_cnt, 1.0)) if pos_loss_sum is not None else float('nan'),
        'neg_loss_mean': float(neg_loss_sum / max(loss_cnt, 1.0)) if neg_loss_sum is not None else float('nan'),
        'pnp_eval_cnt': int(pnp_eval),
        'pnp_ok_cnt': int(pnp_ok),
        'pnp_fail_cnt': int(pnp_fail),
        'pnp_fail_pct': float(pnp_fail_pct),
        'pnp_ransac_eval_cnt': int(ransac_eval),
        'pnp_ransac_ok_cnt': int(ransac_ok),
        'pnp_ransac_fail_cnt': int(max(ransac_eval - ransac_ok, 0)),
        'pnp_ransac_inlier_cnt_mean': float(np.mean(ransac_inliers_ok)) if len(ransac_inliers_ok) else float('nan'),
        'pnp_ransac_inlier_cnt_median': float(np.median(ransac_inliers_ok)) if len(ransac_inliers_ok) else float('nan'),
        'eT_ransac_mean': float(np.mean(eT_r_ok)) if len(eT_r_ok) else float('nan'),
        'eT_ransac_median': float(np.median(eT_r_ok)) if len(eT_r_ok) else float('nan'),
        'eR_ransac_mean': float(np.mean(eR_r_ok)) if len(eR_r_ok) else float('nan'),
        'eR_ransac_median': float(np.median(eR_r_ok)) if len(eR_r_ok) else float('nan'),
        'eT_mean': float(np.mean(eT_ok)) if len(eT_ok) else float('nan'),
        'eT_median': float(np.median(eT_ok)) if len(eT_ok) else float('nan'),
        'eR_mean': float(np.mean(eR_ok)) if len(eR_ok) else float('nan'),
        'eR_median': float(np.median(eR_ok)) if len(eR_ok) else float('nan'),
        'speed_raw_mean': float(np.mean(sp_raw_ok)) if len(sp_raw_ok) else float('nan'),
        'speed_thr_mean': float(np.mean(sp_thr_ok)) if len(sp_thr_ok) else float('nan'),
    }
    return out


def _save_debug_image(savefn, image, kp_gt, kp_pr, kp_pnp=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = image.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
    _, h, w = image.shape
    sx = max(float(w - 1), 1.0)
    sy = max(float(h - 1), 1.0)
    xs_gt = kp_gt[0].cpu().numpy() * sx
    ys_gt = kp_gt[1].cpu().numpy() * sy
    xs_pr = kp_pr[0].cpu().numpy() * sx
    ys_pr = kp_pr[1].cpu().numpy() * sy

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.xlim(0, w - 1)
    plt.ylim(h - 1, 0)
    plt.scatter(xs_gt, ys_gt, c='lime', marker='+', label='gt')
    plt.scatter(xs_pr, ys_pr, c='red', marker='x', label='pred')
    if kp_pnp is not None:
        xs_p, ys_p = kp_pnp
        plt.scatter(xs_p, ys_p, c='cyan', marker='o', s=10, label='pnp_proj')
    plt.axis('off')
    plt.legend(loc='lower right')
    os.makedirs(osp.dirname(savefn), exist_ok=True)
    plt.savefig(savefn, bbox_inches='tight', pad_inches=0)
    plt.close()


def _save_pred_heatmap_overlay(savefn, image, pred_hm, kp_gt=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    img = image.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
    _, h, w = image.shape
    sx = max(float(w - 1), 1.0)
    sy = max(float(h - 1), 1.0)

    hm = pred_hm.detach().cpu()
    hm_max = hm.max(dim=0).values
    hm_max = hm_max / max(float(hm_max.max().item()), 1e-12)
    hm_up = F.interpolate(hm_max.unsqueeze(0).unsqueeze(0), size=(int(h), int(w)), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(hm_up.numpy(), cmap='jet', alpha=0.45, vmin=0.0, vmax=1.0)
    plt.xlim(0, w - 1)
    plt.ylim(h - 1, 0)
    if kp_gt is not None:
        xs_gt = kp_gt[0].cpu().numpy() * sx
        ys_gt = kp_gt[1].cpu().numpy() * sy
        plt.scatter(xs_gt, ys_gt, c='lime', marker='+', label='gt')
        plt.legend(loc='lower right')
    plt.axis('off')
    os.makedirs(osp.dirname(savefn), exist_ok=True)
    plt.savefig(savefn, bbox_inches='tight', pad_inches=0)
    plt.close()


def _save_joint_overlay(savefn, image, kp_gt, kp_pr, pred_hm, gt_hm, bbox=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    img = image.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
    _, h, w = image.shape
    sx = max(float(w - 1), 1.0)
    sy = max(float(h - 1), 1.0)

    xs_gt = kp_gt[0].cpu().numpy() * sx
    ys_gt = kp_gt[1].cpu().numpy() * sy
    xs_pr = kp_pr[0].cpu().numpy() * sx
    ys_pr = kp_pr[1].cpu().numpy() * sy

    phm = pred_hm.detach().cpu()
    phm_max = phm.max(dim=0).values
    phm_max = phm_max / max(float(phm_max.max().item()), 1e-12)
    phm_up = F.interpolate(phm_max.unsqueeze(0).unsqueeze(0), size=(int(h), int(w)), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    ghm = gt_hm.detach().cpu()
    ghm_max = ghm.max(dim=0).values
    ghm_max = ghm_max / max(float(ghm_max.max().item()), 1e-12)
    ghm_up = F.interpolate(ghm_max.unsqueeze(0).unsqueeze(0), size=(int(h), int(w)), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(ghm_up.numpy(), cmap='Greens', alpha=0.30, vmin=0.0, vmax=1.0)
    plt.imshow(phm_up.numpy(), cmap='Reds', alpha=0.30, vmin=0.0, vmax=1.0)
    plt.xlim(0, w - 1)
    plt.ylim(h - 1, 0)
    plt.scatter(xs_gt, ys_gt, c='lime', marker='+', label='gt')
    plt.scatter(xs_pr, ys_pr, c='red', marker='x', label='pred')
    if bbox is not None:
        bb = np.array(bbox, dtype=np.float32).reshape(-1)
        if bb.size >= 4 and np.isfinite(bb[:4]).all():
            plt.title(f'img={w}x{h} hm={int(pred_hm.shape[-1])}x{int(pred_hm.shape[-2])} bb={bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f},{bb[3]:.1f}')
    plt.axis('off')
    plt.legend(loc='lower right')
    os.makedirs(osp.dirname(savefn), exist_ok=True)
    plt.savefig(savefn, bbox_inches='tight', pad_inches=0)
    plt.close()

def _save_heatmap_channels(save_dir, heatmaps, prefix, vmax=1.0):
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hm = heatmaps.detach().cpu()
    K = int(hm.shape[0])
    os.makedirs(save_dir, exist_ok=True)
    for k in range(K):
        plt.figure(figsize=(3, 3))
        plt.imshow(hm[k].numpy(), cmap='jet', vmin=0.0, vmax=float(vmax))
        plt.axis('off')
        plt.savefig(osp.join(save_dir, f'{prefix}_ch{k:02d}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()


def main():
    ap = argparse.ArgumentParser('Overfit debug for synthetic images')
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--dataroot', type=str, required=True)
    ap.add_argument('--dataname', type=str, default='')
    ap.add_argument('--keypts_3d_model', type=str, default='src/utils/tangoPoints.mat')
    ap.add_argument('--camera_json', type=str, default='camera.json')
    ap.add_argument('--input_shape', nargs='+', type=int, default=(224, 224))
    ap.add_argument('--num_keypoints', type=int, default=11)
    ap.add_argument('--heatmap_size', nargs='+', type=int, default=(56, 56))
    ap.add_argument('--heatmap_sigma', type=float, default=2.0)
    ap.add_argument('--krn_head', type=str, default='heatmap', choices=['direct', 'heatmap'])
    ap.add_argument('--backbone', type=str, default='swin_tiny_patch4_window7_224')
    ap.add_argument('--input_normalize', type=str, default='')
    ap.add_argument('--backbone_pretrained_path', type=str, default='')
    ap.add_argument('--no_backbone_pretrained', action='store_true', default=False)
    ap.add_argument('--savedir', type=str, default='log/overfit_debug')
    ap.add_argument('--domain', type=str, default='synthetic')
    ap.add_argument('--csv', type=str, default='splits_krn/train.csv')
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--freeze_iters', type=int, default=200)
    ap.add_argument('--vis_every', type=int, default=100)
    ap.add_argument('--one_batch', action='store_true', default=False)
    ap.add_argument('--save_heatmap_every', type=int, default=50)
    ap.add_argument('--num_samples', type=int, default=32)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--eval_all_overfit_samples', action='store_true', default=False)
    ap.add_argument('--save_pnp_debug', action='store_true', default=False)
    ap.add_argument('--disable_aug', action='store_true', default=False)
    ap.add_argument('--geom_valid_mask', action='store_true', default=False)
    ap.add_argument('--z_min', type=float, default=0.01)
    ap.add_argument('--foreground_valid_mask', action='store_true', default=False)
    ap.add_argument('--fg_thr', type=float, default=0.12)
    ap.add_argument('--fg_patch', type=int, default=3)
    ap.add_argument('--lr_backbone', type=float, default=1e-5)
    ap.add_argument('--lr_head', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--grad_clip_norm', type=float, default=1.0)
    args = ap.parse_args()

    class _Cfg: pass
    cfg = _Cfg()
    cfg.seed = int(args.seed)
    cfg.dataroot = osp.abspath(args.dataroot)
    cfg.dataname = args.dataname
    cfg.model_name = 'krn'
    cfg.dann = False
    cfg.krn_head = args.krn_head
    cfg.num_keypoints = int(args.num_keypoints)
    cfg.num_classes = 5000
    cfg.num_neighbors = 5
    cfg.input_shape = (int(args.input_shape[0]), int(args.input_shape[1]))
    cfg.backbone = args.backbone
    cfg.backbone_pretrained = (not bool(args.no_backbone_pretrained))
    cfg.backbone_pretrained_path = args.backbone_pretrained_path
    if args.input_normalize != '':
        cfg.input_normalize = args.input_normalize
    else:
        cfg.input_normalize = 'none' if cfg.backbone in ('mobilenet_v2', 'alexnet_bvlc', 'bvlc_alexnet') else 'imagenet'
    cfg.debug_shapes = False
    cfg.heatmap_size = (int(args.heatmap_size[0]), int(args.heatmap_size[1]))
    cfg.heatmap_sigma = float(args.heatmap_sigma)
    cfg.lr_backbone = float(args.lr_backbone)
    cfg.lr_head = float(args.lr_head)
    cfg.weight_decay = float(args.weight_decay)
    cfg.grad_clip_norm = float(args.grad_clip_norm)
    cfg.use_cuda = torch.cuda.is_available()
    cfg.geom_valid_mask = bool(args.geom_valid_mask)
    cfg.z_min = float(args.z_min)

    device = torch.device('cuda:0') if cfg.use_cuda else torch.device('cpu')
    set_all_seeds(cfg.seed, cfg, cfg.use_cuda)

    root, csv_path = _resolve_root_and_csv(cfg.dataroot, cfg.dataname, args.domain, args.csv)
    tfm = build_transforms('krn', cfg.input_shape, is_train=False)

    ds_full = OverfitDataset(root, csv_path, num_keypoints=cfg.num_keypoints, transforms=tfm, limit=0)
    rng = np.random.RandomState(int(cfg.seed))
    idxs = rng.choice(len(ds_full), size=min(int(args.num_samples), len(ds_full)), replace=False).tolist()
    ds = torch.utils.data.Subset(ds_full, idxs)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=not bool(args.one_batch), num_workers=0, drop_last=True)
    dl_eval = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)

    model = get_model(cfg).to(device)

    from torch.optim import AdamW
    if bool(args.one_batch):
        optim = AdamW(
            [{'params': [p for n, p in model.named_parameters() if (not n.startswith('backbone.'))], 'lr': float(cfg.lr_head)}],
            weight_decay=float(cfg.weight_decay),
        )
        if hasattr(model, 'set_backbone_trainable'):
            model.set_backbone_trainable(False)
    else:
        optim = AdamW(
            [
                {'params': [p for n, p in model.named_parameters() if n.startswith('backbone.')], 'lr': float(cfg.lr_backbone)},
                {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone.')], 'lr': float(cfg.lr_head)},
            ],
            weight_decay=float(cfg.weight_decay),
        )

    kp3d_path = osp.join(osp.dirname(__file__), args.keypts_3d_model) if not osp.isabs(args.keypts_3d_model) else args.keypts_3d_model
    cam_path = osp.join(root, args.camera_json) if not osp.isabs(args.camera_json) else args.camera_json
    corners3D = load_tango_3d_keypoints(kp3d_path)
    cameraMatrix, distCoeffs = load_camera_intrinsics(cam_path)

    it = 0
    model.train()
    fixed_batch = None
    if bool(args.one_batch):
        for images, bbox, keypts, q_gt, t_gt in dl:
            fixed_batch = (images, bbox, keypts, q_gt, t_gt)
            break
        assert fixed_batch is not None

    best_rmse = float('inf')
    best_it = -1
    os.makedirs(args.savedir, exist_ok=True)
    with open(osp.join(args.savedir, 'overfit_indices.txt'), 'w', encoding='utf-8') as f:
        f.write(','.join(str(i) for i in idxs) + '\n')

    while it < int(args.iters):
        for images, bbox, keypts, q_gt, t_gt in ([fixed_batch] if bool(args.one_batch) else dl):
            it += 1
            if it > int(args.iters):
                break

            if (not bool(args.one_batch)) and hasattr(model, 'set_backbone_trainable'):
                model.set_backbone_trainable(it > int(args.freeze_iters))
            images = images.to(device)
            keypts = keypts.to(device)
            bbox = bbox.cpu()

            if bool(getattr(cfg, 'geom_valid_mask', False)):
                key_m = torch.isfinite(keypts[:, 0, :]) & torch.isfinite(keypts[:, 1, :]) & (keypts[:, 0, :] >= 0) & (keypts[:, 0, :] <= 1) & (keypts[:, 1, :] >= 0) & (keypts[:, 1, :] <= 1)
                for b in range(images.shape[0]):
                    vm = key_m[b]
                    if bool(getattr(cfg, 'geom_valid_mask', False)):
                        gm = geom_valid_mask(
                            q_gt[b].detach().cpu().numpy(),
                            t_gt[b].detach().cpu().numpy(),
                            bbox[b].detach().cpu().numpy(),
                            corners3D,
                            cameraMatrix,
                            distCoeffs,
                            heatmap_size=cfg.heatmap_size,
                            z_min=float(getattr(cfg, 'z_min', 1e-6)),
                        )
                        vm = vm & torch.as_tensor(gm, dtype=torch.bool, device=vm.device)
                    if int(vm.sum().detach().cpu()) < int(key_m[b].sum().detach().cpu()):
                        bad = ~vm
                        keypts[b, :, bad] = float('nan')

            loss, sm = model(images, keypts)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if it <= 10:
                head_g2 = 0.0
                bb_g2 = 0.0
                head_p = 0
                bb_p = 0
                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if n.startswith('backbone.'):
                        bb_p += 1
                    else:
                        head_p += 1
                    if p.grad is None:
                        continue
                    g2 = float(p.grad.detach().float().pow(2).sum().cpu())
                    if n.startswith('backbone.'):
                        bb_g2 += g2
                    else:
                        head_g2 += g2
                head_g = head_g2 ** 0.5
                bb_g = bb_g2 ** 0.5
                pg_sizes = [len(pg.get('params', [])) for pg in optim.param_groups]
                print(
                    f'iter={it:04d} loss_hm={sm.get("loss_hm", sm.get("loss_x", float("nan"))):.6f} '
                    f'pred_logits(min/max/mean/std)='
                    f'{sm.get("pred_logits_min", float("nan")):.4f}/'
                    f'{sm.get("pred_logits_max", float("nan")):.4f}/'
                    f'{sm.get("pred_logits_mean", float("nan")):.4f}/'
                    f'{sm.get("pred_logits_std", float("nan")):.4f} '
                    f'pred_hm(min/max/mean/std)='
                    f'{sm.get("pred_hm_min", float("nan")):.4f}/'
                    f'{sm.get("pred_hm_max", float("nan")):.4f}/'
                    f'{sm.get("pred_hm_mean", float("nan")):.4f}/'
                    f'{sm.get("pred_hm_std", float("nan")):.4f} '
                    f'gt_hm(min/max/mean/std)='
                    f'{sm.get("gt_hm_min", float("nan")):.4f}/'
                    f'{sm.get("gt_hm_max", float("nan")):.4f}/'
                    f'{sm.get("gt_hm_mean", float("nan")):.4f}/'
                    f'{sm.get("gt_hm_std", float("nan")):.4f} '
                    f'head_grad_norm={head_g:.6f} backbone_grad_norm={bb_g:.6f} '
                    f'head_params={head_p} backbone_params={bb_p} opt_groups={pg_sizes}'
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(cfg, 'grad_clip_norm', 1.0)))
            optim.step()

            rmse_px = sm.get("rmse_px", float("nan"))
            inside = sm.get("inside_pct", float("nan"))
            if not np.isfinite(rmse_px) or not np.isfinite(inside):
                try:
                    model.eval()
                    with torch.no_grad():
                        if hasattr(model, 'last_pred_heatmaps') and model.last_pred_heatmaps is not None:
                            pred_hm = model.last_pred_heatmaps.detach()
                            pred_kp, _ = heatmap_to_keypoints(pred_hm, decode=getattr(cfg, 'heatmap_decode', 'argmax'), beta=float(getattr(cfg, 'heatmap_beta', 100.0)))
                        else:
                            xc, yc = model(images)
                            pred_kp = torch.stack([xc.to(device), yc.to(device)], dim=1)
                    valid = torch.isfinite(keypts[:, 0, :]) & torch.isfinite(keypts[:, 1, :]) & (keypts[:, 0, :] >= 0) & (keypts[:, 0, :] <= 1) & (keypts[:, 1, :] >= 0) & (keypts[:, 1, :] <= 1)
                    rmse_px = float(keypoint_rmse_pixels(pred_kp, keypts, image_size=cfg.input_shape, valid_mask=valid).detach().cpu())
                    inside = float(inside_image_percentage(pred_kp, valid).detach().cpu())
                finally:
                    model.train()

            if it % 20 == 0:
                print(
                    f'iter={it} loss={sm.get("loss_hm", sm["loss_x"]):.6f} train_batch_rmse_px={rmse_px:.3f} inside={inside:.2f} '
                    f'peak_mean={sm.get("heatmap_peak_mean", float("nan")):.6f} ent={sm.get("heatmap_entropy", float("nan")):.6f} ent_beta={sm.get("heatmap_entropy_beta", float("nan")):.6f} '
                    f'coll_dist={sm.get("collapsed_keypoint_distance", float("nan")):.3f}'
                )

            if bool(args.eval_all_overfit_samples) and (int(args.eval_every) > 0) and (it % int(args.eval_every) == 0):
                metrics = _eval_overfit(
                    model=model,
                    dl_eval=dl_eval,
                    device=device,
                    cfg=cfg,
                    corners3D=corners3D,
                    cameraMatrix=cameraMatrix,
                    distCoeffs=distCoeffs,
                    it=it,
                    savedir=osp.join(args.savedir, f'eval_iter{it:05d}') if bool(args.save_pnp_debug) else args.savedir,
                    save_debug=bool(args.save_pnp_debug),
                )
                print(
                    f'overfit_eval_count={metrics["overfit_eval_count"]} '
                    f'eval_rmse_px_mean={metrics["rmse_px_mean"]:.3f} eval_rmse_px_median={metrics["rmse_px_median"]:.3f} '
                    f'pnp_eval_cnt={metrics["pnp_eval_cnt"]} pnp_ok_cnt={metrics["pnp_ok_cnt"]} pnp_fail_cnt={metrics["pnp_fail_cnt"]} pnp_fail_pct={metrics["pnp_fail_pct"]:.2f} '
                    f'ransac_ok/inl_mean/med={metrics["pnp_ransac_ok_cnt"]}/{metrics["pnp_ransac_inlier_cnt_mean"]:.2f}/{metrics["pnp_ransac_inlier_cnt_median"]:.2f} '
                    f'eT_mean/med={metrics["eT_mean"]:.3f}/{metrics["eT_median"]:.3f} eR_mean/med={metrics["eR_mean"]:.3f}/{metrics["eR_median"]:.3f} '
                    f'spd_raw/thr={metrics["speed_raw_mean"]:.3f}/{metrics["speed_thr_mean"]:.3f} '
                    f'pos_loss_mean={metrics["pos_loss_mean"]:.6f} neg_loss_mean={metrics["neg_loss_mean"]:.6f} '
                    f'worst5={metrics.get("rmse_worst5", [])}'
                )
                if metrics['rmse_px_mean'] < best_rmse:
                    best_rmse = float(metrics['rmse_px_mean'])
                    best_it = int(it)
                    torch.save(model.state_dict(), osp.join(args.savedir, 'checkpoint_overfit_best.pth'))
                    if bool(args.save_pnp_debug):
                        pass
                    metrics_best_dir = osp.join(args.savedir, f'best_iter{best_it:05d}_rmse{best_rmse:.3f}')
                    _ = _eval_overfit(
                        model=model,
                        dl_eval=dl_eval,
                        device=device,
                        cfg=cfg,
                        corners3D=corners3D,
                        cameraMatrix=cameraMatrix,
                        distCoeffs=distCoeffs,
                        it=best_it,
                        savedir=metrics_best_dir,
                        save_debug=True,
                    )
                    with open(osp.join(args.savedir, 'overfit_best.txt'), 'w', encoding='utf-8') as f:
                        f.write(f'best_it={best_it}\n')
                        f.write(f'best_rmse_px_mean={best_rmse}\n')
                        for k, v in metrics.items():
                            if isinstance(v, list):
                                f.write(f'{k}={v}\n')
                            else:
                                f.write(f'{k}={v}\n')

            if bool(args.one_batch) and (it % int(args.save_heatmap_every) == 0):
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'last_pred_heatmaps') and model.last_pred_heatmaps is not None:
                        pred_hm = model.last_pred_heatmaps.detach().cpu()
                    else:
                        pred_hm = None
                if pred_hm is not None:
                    for b in range(min(4, images.shape[0])):
                        _save_pred_heatmap_overlay(
                            osp.join(args.savedir, f'iter{it:05d}_b{b}_predhm.png'),
                            images[b].detach().cpu(),
                            pred_hm[b],
                            kp_gt=keypts[b].detach().cpu(),
                        )
                model.train()

            if it % int(args.vis_every) == 0:
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'last_pred_heatmaps') and model.last_pred_heatmaps is not None:
                        pred_hm = model.last_pred_heatmaps.detach()
                        pred_kp, _ = heatmap_to_keypoints(pred_hm, decode=getattr(cfg, 'heatmap_decode', 'argmax'), beta=float(getattr(cfg, 'heatmap_beta', 100.0)))
                    else:
                        xc, yc = model(images)
                        pred_kp = torch.stack([xc.to(device), yc.to(device)], dim=1)
                for b in range(min(4, images.shape[0])):
                    kp_gt = keypts[b].detach().cpu()
                    kp_pr = pred_kp[b].detach().cpu()
                    try:
                        corners2D = torch.stack([kp_pr[0], kp_pr[1]], dim=0)
                        corners2D = corners2D.t().numpy()
                        bb = bbox[b].numpy()
                        corners2D[:, 0] = corners2D[:, 0] * (bb[1] - bb[0]) + bb[0]
                        corners2D[:, 1] = corners2D[:, 1] * (bb[3] - bb[2]) + bb[2]
                        q_pr, t_pr = pnp(corners3D, corners2D, cameraMatrix, distCoeffs)
                        pts2d_proj = project_keypoints(q_pr, t_pr, cameraMatrix, distCoeffs, corners3D.T)
                        _, h, w = images[b].shape
                        sx = max(float(w - 1), 1.0)
                        sy = max(float(h - 1), 1.0)
                        dx = max(float(bb[1] - bb[0]), 1.0)
                        dy = max(float(bb[3] - bb[2]), 1.0)
                        xs_p = (pts2d_proj[0] - float(bb[0])) / dx * sx
                        ys_p = (pts2d_proj[1] - float(bb[2])) / dy * sy
                        _save_debug_image(
                            osp.join(args.savedir, f'iter{it:05d}_b{b}.png'),
                            images[b].detach().cpu(),
                            kp_gt,
                            kp_pr,
                            kp_pnp=(xs_p, ys_p),
                        )
                    except Exception:
                        _save_debug_image(
                            osp.join(args.savedir, f'iter{it:05d}_b{b}_pnp_fail.png'),
                            images[b].detach().cpu(),
                            kp_gt,
                            kp_pr,
                        )
                model.train()

    metrics = _eval_overfit(
        model=model,
        dl_eval=dl_eval,
        device=device,
        cfg=cfg,
        corners3D=corners3D,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        it=int(args.iters),
        savedir=osp.join(args.savedir, f'final_iter{int(args.iters):05d}') if bool(args.save_pnp_debug) else args.savedir,
        save_debug=bool(args.save_pnp_debug),
    )
    print(
        f'overfit_eval_count={metrics["overfit_eval_count"]} '
        f'eval_rmse_px_mean={metrics["rmse_px_mean"]:.3f} eval_rmse_px_median={metrics["rmse_px_median"]:.3f} '
        f'pnp_eval_cnt={metrics["pnp_eval_cnt"]} pnp_ok_cnt={metrics["pnp_ok_cnt"]} pnp_fail_cnt={metrics["pnp_fail_cnt"]} pnp_fail_pct={metrics["pnp_fail_pct"]:.2f} '
        f'ransac_ok/inl_mean/med={metrics["pnp_ransac_ok_cnt"]}/{metrics["pnp_ransac_inlier_cnt_mean"]:.2f}/{metrics["pnp_ransac_inlier_cnt_median"]:.2f} '
        f'eT_mean/med={metrics["eT_mean"]:.3f}/{metrics["eT_median"]:.3f} eR_mean/med={metrics["eR_mean"]:.3f}/{metrics["eR_median"]:.3f} '
        f'spd_raw/thr={metrics["speed_raw_mean"]:.3f}/{metrics["speed_thr_mean"]:.3f} '
        f'pos_loss_mean={metrics["pos_loss_mean"]:.6f} neg_loss_mean={metrics["neg_loss_mean"]:.6f} '
        f'worst5={metrics.get("rmse_worst5", [])}'
    )


if __name__ == '__main__':
    main()
