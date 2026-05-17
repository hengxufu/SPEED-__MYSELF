import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.datasets.transforms import build_transforms
from src.utils.heatmaps import gaussian_heatmaps_from_keypoints, heatmap_to_keypoints_argmax, heatmap_to_keypoints_softargmax
from src.utils.heatmaps import keypoint_rmse_pixels, inside_image_percentage, per_keypoint_error_pixels
from src.utils.utils import load_tango_3d_keypoints, load_camera_intrinsics, pnp, project_keypoints
from src.utils.metrics import error_orientation, error_translation, speed_score


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


class _EvalDataset(Dataset):
    def __init__(self, root, csv_path, num_keypoints=11, transforms=None, limit=200):
        self.root = root
        self.num_keypoints = int(num_keypoints)
        self.transforms = transforms
        self.df = pd.read_csv(csv_path, header=None).iloc[:int(limit)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_numpy()
        imgpath = osp.join(self.root, row[0])
        img = Image.open(imgpath).convert('RGB')
        bbox = np.array(row[1:5], dtype=np.float32)
        q_gt = np.array(row[5:9], dtype=np.float32)
        t_gt = np.array(row[9:12], dtype=np.float32)
        keypts = np.array(row[12:], dtype=np.float32).reshape(self.num_keypoints, 2).T
        if self.transforms is not None:
            img, bbox, keypts = self.transforms(img, bbox, keypts)
        return img, torch.as_tensor(bbox, dtype=torch.float32), torch.as_tensor(keypts, dtype=torch.float32), torch.as_tensor(q_gt), torch.as_tensor(t_gt)


def _pnp_from_norm_keypoints(keypts_norm, bbox, corners3D, cameraMatrix, distCoeffs, valid_mask=None):
    kp = keypts_norm.detach().cpu().numpy().T
    bb = bbox.detach().cpu().numpy()
    xmin, xmax, ymin, ymax = bb
    kp[:, 0] = kp[:, 0] * (xmax - xmin) + xmin
    kp[:, 1] = kp[:, 1] * (ymax - ymin) + ymin
    if valid_mask is not None:
        m = valid_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        if int(m.sum()) >= 4:
            kp = kp[m]
            pts3d = corners3D[m]
        else:
            kp = kp
            pts3d = corners3D
    else:
        pts3d = corners3D
    spread = np.ptp(kp, axis=0)
    if (not np.isfinite(spread).all()) or float(spread.min()) < 2.0:
        raise ValueError('Degenerate 2D keypoints for PnP')
    q_pr, t_pr = pnp(pts3d, kp, cameraMatrix, distCoeffs)
    if (not np.isfinite(q_pr).all()) or (not np.isfinite(t_pr).all()):
        raise ValueError('Non-finite pose from PnP')
    if float(np.linalg.norm(t_pr)) > 200.0:
        raise ValueError('Implausible translation from PnP')
    return q_pr, t_pr, kp


def _save_vis(savefn, image, kp_gt, kp_dec, bbox, q_pr, t_pr, corners3D, cameraMatrix, distCoeffs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = image.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
    _, h, w = image.shape
    sx = max(float(w - 1), 1.0)
    sy = max(float(h - 1), 1.0)
    xs_gt = kp_gt[0].cpu().numpy() * sx
    ys_gt = kp_gt[1].cpu().numpy() * sy
    xs_dec = kp_dec[0].cpu().numpy() * sx
    ys_dec = kp_dec[1].cpu().numpy() * sy

    pts2d_proj = None
    try:
        pts2d = project_keypoints(q_pr, t_pr, cameraMatrix, distCoeffs, corners3D.T)
        bb = bbox.cpu().numpy()
        dx = max(float(bb[1] - bb[0]), 1.0)
        dy = max(float(bb[3] - bb[2]), 1.0)
        xs_p = (pts2d[0] - float(bb[0])) / dx * sx
        ys_p = (pts2d[1] - float(bb[2])) / dy * sy
        pts2d_proj = (xs_p, ys_p)
    except Exception:
        pts2d_proj = None

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.xlim(0, w - 1)
    plt.ylim(h - 1, 0)
    plt.scatter(xs_gt, ys_gt, c='lime', marker='+', label='gt')
    plt.scatter(xs_dec, ys_dec, c='red', marker='x', label='decoded')
    if pts2d_proj is not None:
        plt.scatter(pts2d_proj[0], pts2d_proj[1], c='cyan', marker='o', s=10, label='pnp_proj')
    plt.axis('off')
    plt.legend(loc='lower right')
    os.makedirs(osp.dirname(savefn), exist_ok=True)
    plt.savefig(savefn, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser('Stage-1: verify GT heatmap label and decoding loop')
    ap.add_argument('--dataroot', type=str, required=True)
    ap.add_argument('--dataname', type=str, default='')
    ap.add_argument('--domain', type=str, default='synthetic')
    ap.add_argument('--csv', type=str, default='splits_krn/validation.csv')
    ap.add_argument('--input_shape', nargs='+', type=int, default=(224, 224))
    ap.add_argument('--heatmap_size', nargs='+', type=int, default=(56, 56))
    ap.add_argument('--heatmap_sigma', type=float, default=2.0)
    ap.add_argument('--decode', type=str, default='argmax', choices=['argmax', 'softargmax'])
    ap.add_argument('--beta', type=float, default=100.0)
    ap.add_argument('--num_keypoints', type=int, default=11)
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--savedir', type=str, default='log/stage1_heatmap_label_decode')
    ap.add_argument('--save_n', type=int, default=20)
    args = ap.parse_args()

    root, csv_path = _resolve_root_and_csv(args.dataroot, args.dataname, args.domain, args.csv)

    tfm = build_transforms('krn', (int(args.input_shape[0]), int(args.input_shape[1])), p_aug=0.0, is_train=False)
    ds = _EvalDataset(root, csv_path, num_keypoints=int(args.num_keypoints), transforms=tfm, limit=int(args.num))
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)

    kp3d_path = osp.join(osp.dirname(__file__), 'src/utils/tangoPoints.mat')
    cam_path = osp.join(root, 'camera.json')
    corners3D = load_tango_3d_keypoints(kp3d_path)
    cameraMatrix, distCoeffs = load_camera_intrinsics(cam_path)

    rmses, insides, eTs_gt, eRs_gt, eTs_dec, eRs_dec = [], [], [], [], [], []
    per_kp_err_all = []
    saved = 0

    for images, bbox, keypts, q_gt, t_gt in dl:
        gt_hm, valid = gaussian_heatmaps_from_keypoints(keypts, (int(args.heatmap_size[0]), int(args.heatmap_size[1])), sigma=float(args.heatmap_sigma))
        if args.decode == 'softargmax':
            dec_kp, _ = heatmap_to_keypoints_softargmax(gt_hm, beta=float(args.beta))
        else:
            dec_kp, _ = heatmap_to_keypoints_argmax(gt_hm)

        rmse = keypoint_rmse_pixels(dec_kp, keypts, image_size=(int(args.input_shape[0]), int(args.input_shape[1])), valid_mask=valid)
        inside = inside_image_percentage(dec_kp, valid)
        rmses.append(float(rmse.detach().cpu()))
        insides.append(float(inside.detach().cpu()))

        per_err = per_keypoint_error_pixels(dec_kp, keypts, image_size=(int(args.input_shape[0]), int(args.input_shape[1]))).detach().cpu()
        per_kp_err_all.append(per_err.numpy())

        B = images.shape[0]
        for b in range(B):
            vm = valid[b]
            q_gt_pnp = np.array([1, 0, 0, 0], dtype=np.float32)
            t_gt_pnp = np.array([0, 0, 0], dtype=np.float32)
            try:
                q_pr, t_pr, _ = _pnp_from_norm_keypoints(keypts[b], bbox[b], corners3D, cameraMatrix, distCoeffs, valid_mask=vm)
                q_gt_pnp = q_pr
                t_gt_pnp = t_pr
                eR = error_orientation(q_pr, q_gt[b].numpy())
                eT = error_translation(t_pr, t_gt[b].numpy())
                eRs_gt.append(float(eR))
                eTs_gt.append(float(eT))
            except Exception:
                eRs_gt.append(float('nan'))
                eTs_gt.append(float('nan'))

            q_dec_pnp = np.array([1, 0, 0, 0], dtype=np.float32)
            t_dec_pnp = np.array([0, 0, 0], dtype=np.float32)
            try:
                q_pr, t_pr, _ = _pnp_from_norm_keypoints(dec_kp[b], bbox[b], corners3D, cameraMatrix, distCoeffs, valid_mask=vm)
                q_dec_pnp = q_pr
                t_dec_pnp = t_pr
                eR = error_orientation(q_pr, q_gt[b].numpy())
                eT = error_translation(t_pr, t_gt[b].numpy())
                eRs_dec.append(float(eR))
                eTs_dec.append(float(eT))
            except Exception:
                eRs_dec.append(float('nan'))
                eTs_dec.append(float('nan'))

            if saved < int(args.save_n):
                tag = f'idx{saved:05d}_rmse{float(rmse.detach().cpu()):.3f}_eT{eTs_dec[-1]:.3f}_eR{eRs_dec[-1]:.3f}'
                _save_vis(
                    osp.join(args.savedir, tag + '.png'),
                    images[b].detach().cpu(),
                    keypts[b].detach().cpu(),
                    dec_kp[b].detach().cpu(),
                    bbox[b].detach().cpu(),
                    q_dec_pnp,
                    t_dec_pnp,
                    corners3D,
                    cameraMatrix,
                    distCoeffs,
                )
                saved += 1

    per_kp_err_all = np.concatenate(per_kp_err_all, axis=0) if len(per_kp_err_all) > 0 else np.zeros((0, int(args.num_keypoints)), dtype=np.float32)
    rmse_mean = float(np.nanmean(rmses)) if len(rmses) else float('nan')
    rmse_med = float(np.nanmedian(rmses)) if len(rmses) else float('nan')
    eT_gt_mean = float(np.nanmean(eTs_gt)) if len(eTs_gt) else float('nan')
    eR_gt_mean = float(np.nanmean(eRs_gt)) if len(eRs_gt) else float('nan')
    eT_dec_mean = float(np.nanmean(eTs_dec)) if len(eTs_dec) else float('nan')
    eR_dec_mean = float(np.nanmean(eRs_dec)) if len(eRs_dec) else float('nan')

    print('csv=', csv_path)
    print('num_samples=', len(ds))
    print('heatmap_size=', tuple(int(x) for x in args.heatmap_size), 'sigma=', float(args.heatmap_sigma), 'decode=', args.decode)
    print('decoded_rmse_px mean/median=', rmse_mean, rmse_med)
    if per_kp_err_all.shape[0] > 0:
        per_kp_mean = np.nanmean(per_kp_err_all, axis=0)
        print('per_keypoint_err_px mean=', per_kp_mean.tolist())
    print('GT->PnP eT/eR mean=', eT_gt_mean, eR_gt_mean)
    print('Decoded->PnP eT/eR mean=', eT_dec_mean, eR_dec_mean)
    print('savedir=', osp.abspath(args.savedir))


if __name__ == '__main__':
    main()
