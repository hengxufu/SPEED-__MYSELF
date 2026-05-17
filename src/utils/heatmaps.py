from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch


def gaussian_heatmaps_from_keypoints(keypts_norm, heatmap_size, sigma=2.0):
    """
    Args:
        keypts_norm: (B, 2, K) in [0,1] relative to cropped/resized ROI
        heatmap_size: (H, W)
        sigma: gaussian std in heatmap pixels
    Returns:
        heatmaps: (B, K, H, W)
        valid_mask: (B, K) boolean
    """
    assert keypts_norm.dim() == 3 and keypts_norm.shape[1] == 2
    B, _, K = keypts_norm.shape
    H, W = int(heatmap_size[0]), int(heatmap_size[1])
    device = keypts_norm.device

    xs = keypts_norm[:, 0, :]
    ys = keypts_norm[:, 1, :]

    valid = torch.isfinite(xs) & torch.isfinite(ys) & (xs >= 0) & (xs <= 1) & (ys >= 0) & (ys <= 1)
    xh = xs * (W - 1)
    yh = ys * (H - 1)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)

    xh = xh.view(B, K, 1, 1)
    yh = yh.view(B, K, 1, 1)

    s2 = float(sigma) ** 2
    s2 = s2 if s2 > 1e-12 else 1e-12
    heat = torch.exp(-((xx - xh) ** 2 + (yy - yh) ** 2) / (2.0 * s2))

    mask = valid.to(dtype=heat.dtype).view(B, K, 1, 1)
    heat = heat * mask
    return heat, valid


def per_keypoint_error_pixels(pred_norm, gt_norm, image_size):
    assert pred_norm.dim() == 3 and gt_norm.dim() == 3
    H, W = float(image_size[0]), float(image_size[1])
    dx = (pred_norm[:, 0, :] - gt_norm[:, 0, :]) * (W - 1.0)
    dy = (pred_norm[:, 1, :] - gt_norm[:, 1, :]) * (H - 1.0)
    return torch.sqrt(dx ** 2 + dy ** 2)


def per_keypoint_rmse_pixels(pred_norm, gt_norm, image_size, valid_mask):
    assert pred_norm.dim() == 3 and gt_norm.dim() == 3 and valid_mask.dim() == 2
    H, W = float(image_size[0]), float(image_size[1])
    dx = (pred_norm[:, 0, :] - gt_norm[:, 0, :]) * (W - 1.0)
    dy = (pred_norm[:, 1, :] - gt_norm[:, 1, :]) * (H - 1.0)
    se = dx ** 2 + dy ** 2
    m = valid_mask.to(dtype=se.dtype)
    denom = torch.clamp(m.sum(dim=0), min=1.0)
    mse = (se * m).sum(dim=0) / denom
    return torch.sqrt(mse)


def heatmap_to_keypoints_argmax(heatmaps):
    """
    Args:
        heatmaps: (B, K, H, W)
    Returns:
        keypts_norm: (B, 2, K) normalized to [0,1] in heatmap/image coordinates
        conf: (B, K) max heat value
    """
    assert heatmaps.dim() == 4
    heatmaps = torch.nan_to_num(heatmaps, nan=0.0, posinf=0.0, neginf=0.0)
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    idx = torch.argmax(flat, dim=-1)  # (B,K)
    conf = torch.gather(flat, -1, idx.unsqueeze(-1)).squeeze(-1)
    xs = (idx % W).to(dtype=torch.float32)
    ys = (idx // W).to(dtype=torch.float32)
    xs = xs / max(float(W - 1), 1.0)
    ys = ys / max(float(H - 1), 1.0)
    keypts = torch.stack([xs, ys], dim=1)  # (B,2,K)
    return keypts, conf


def heatmap_to_keypoints_softargmax(heatmaps, beta=100.0):
    """
    Args:
        heatmaps: (B, K, H, W)
        beta: temperature for softmax; larger -> closer to argmax
    Returns:
        keypts_norm: (B, 2, K) normalized to [0,1]
        conf: (B, K) max heat value (raw)
    """
    assert heatmaps.dim() == 4
    heatmaps = torch.nan_to_num(heatmaps, nan=0.0, posinf=0.0, neginf=0.0)
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    conf = flat.max(dim=-1).values
    p = torch.softmax(flat * float(beta), dim=-1).view(B, K, H, W)

    xs = torch.arange(W, device=heatmaps.device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.arange(H, device=heatmaps.device, dtype=torch.float32).view(1, 1, H, 1)
    ex = (p * xs).sum(dim=(2, 3))  # (B,K)
    ey = (p * ys).sum(dim=(2, 3))  # (B,K)
    ex = ex / max(float(W - 1), 1.0)
    ey = ey / max(float(H - 1), 1.0)
    keypts = torch.stack([ex, ey], dim=1)  # (B,2,K)
    return keypts, conf


def heatmap_to_keypoints_dsnt(heatmaps_prob):
    assert heatmaps_prob.dim() == 4
    hm = torch.nan_to_num(heatmaps_prob, nan=0.0, posinf=0.0, neginf=0.0)
    B, K, H, W = hm.shape
    flat = hm.view(B, K, -1)
    s = flat.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    p = flat / s
    conf = p.max(dim=-1).values

    xs = torch.arange(W, device=hm.device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.arange(H, device=hm.device, dtype=torch.float32).view(1, 1, H, 1)
    p2 = p.view(B, K, H, W)
    ex = (p2 * xs).sum(dim=(2, 3))
    ey = (p2 * ys).sum(dim=(2, 3))
    ex = ex / max(float(W - 1), 1.0)
    ey = ey / max(float(H - 1), 1.0)
    keypts = torch.stack([ex, ey], dim=1)
    return keypts, conf


def keypoint_rmse_pixels(pred_norm, gt_norm, image_size, valid_mask):
    """
    Args:
        pred_norm: (B,2,K) normalized [0,1]
        gt_norm: (B,2,K) normalized [0,1]
        image_size: (H,W) of the ROI input image
        valid_mask: (B,K) boolean
    Returns:
        rmse: scalar float tensor
    """
    H, W = float(image_size[0]), float(image_size[1])
    px = (pred_norm[:, 0, :] - gt_norm[:, 0, :]) * (W - 1.0)
    py = (pred_norm[:, 1, :] - gt_norm[:, 1, :]) * (H - 1.0)
    se = px ** 2 + py ** 2
    m = valid_mask.to(dtype=se.dtype)
    denom = torch.clamp(m.sum(), min=1.0)
    mse = (se * m).sum() / denom
    return torch.sqrt(mse)


def inside_image_percentage(keypts_norm, valid_mask):
    xs = keypts_norm[:, 0, :]
    ys = keypts_norm[:, 1, :]
    inside = (xs >= 0) & (xs <= 1) & (ys >= 0) & (ys <= 1)
    m = valid_mask
    denom = torch.clamp(m.sum(), min=1)
    return (inside & m).sum().to(dtype=torch.float32) * 100.0 / denom.to(dtype=torch.float32)


def heatmap_peaks(heatmaps):
    assert heatmaps.dim() == 4
    hm = torch.nan_to_num(heatmaps, nan=0.0, posinf=0.0, neginf=0.0)
    B, K, H, W = hm.shape
    flat = hm.view(B, K, H * W)
    return flat.max(dim=-1).values


def heatmap_topk_peaks(heatmaps, k=3):
    assert heatmaps.dim() == 4
    hm = torch.nan_to_num(heatmaps, nan=0.0, posinf=0.0, neginf=0.0)
    B, K, H, W = hm.shape
    flat = hm.view(B, K, H * W)
    kk = int(k)
    kk = max(1, min(kk, H * W))
    vals, idx = torch.topk(flat, k=kk, dim=-1, largest=True, sorted=True)  # (B,K,kk)
    xs = (idx % W).to(dtype=torch.float32)
    ys = (idx // W).to(dtype=torch.float32)
    xs = xs / max(float(W - 1), 1.0)
    ys = ys / max(float(H - 1), 1.0)
    coords = torch.stack([xs, ys], dim=2)  # (B,K,2,kk)
    return coords, vals


def heatmap_peak_to_second_ratio(heatmaps, eps=1e-12):
    assert heatmaps.dim() == 4
    _, vals = heatmap_topk_peaks(heatmaps, k=2)
    if vals.shape[-1] < 2:
        return torch.ones_like(vals.squeeze(-1))
    v1 = vals[..., 0]
    v2 = vals[..., 1]
    return v1 / (v2.clamp(min=float(eps)))


def per_keypoint_pck(pred_norm, gt_norm, image_size, valid_mask, thr_px):
    assert pred_norm.dim() == 3 and gt_norm.dim() == 3 and valid_mask.dim() == 2
    err = per_keypoint_error_pixels(pred_norm, gt_norm, image_size=image_size)  # (B,K)
    m = valid_mask.to(dtype=err.dtype)
    ok = (err <= float(thr_px)).to(dtype=err.dtype) * m
    denom = torch.clamp(m.sum(dim=0), min=1.0)
    return ok.sum(dim=0) / denom


def heatmap_entropy(heatmaps, eps=1e-12, beta=1.0):
    assert heatmaps.dim() == 4
    hm = torch.nan_to_num(heatmaps, nan=0.0, posinf=0.0, neginf=0.0)
    B, K, H, W = hm.shape
    flat = hm.view(B, K, H * W)
    p = torch.softmax(flat * float(beta), dim=-1)
    ent = -(p * (p.clamp(min=float(eps)).log())).sum(dim=-1)
    ent = ent / max(float(math.log(H * W)), 1.0)
    return ent


def collapsed_keypoint_distance(keypts_norm, image_size, valid_mask):
    assert keypts_norm.dim() == 3 and valid_mask.dim() == 2
    H, W = float(image_size[0]), float(image_size[1])
    xs = keypts_norm[:, 0, :] * (W - 1.0)
    ys = keypts_norm[:, 1, :] * (H - 1.0)
    m = valid_mask.to(dtype=xs.dtype)
    denom = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
    mx = (xs * m).sum(dim=1, keepdim=True) / denom
    my = (ys * m).sum(dim=1, keepdim=True) / denom
    d = torch.sqrt((xs - mx) ** 2 + (ys - my) ** 2)
    return (d * m).sum(dim=1) / denom.squeeze(1)


def keypoint_spread_min_px(keypts_norm, image_size, valid_mask):
    assert keypts_norm.dim() == 3 and valid_mask.dim() == 2
    H, W = float(image_size[0]), float(image_size[1])
    xs = keypts_norm[:, 0, :] * (W - 1.0)
    ys = keypts_norm[:, 1, :] * (H - 1.0)
    m = valid_mask.to(dtype=xs.dtype)
    big = torch.tensor(1e9, device=xs.device, dtype=xs.dtype)
    x_masked = torch.where(m > 0, xs, big)
    y_masked = torch.where(m > 0, ys, big)
    x_min = x_masked.min(dim=1).values
    y_min = y_masked.min(dim=1).values
    x_max = torch.where(m > 0, xs, -big).max(dim=1).values
    y_max = torch.where(m > 0, ys, -big).max(dim=1).values
    dx = x_max - x_min
    dy = y_max - y_min
    return torch.minimum(dx, dy)
