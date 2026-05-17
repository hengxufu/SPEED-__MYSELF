import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from src.datasets.transforms import build_transforms
from src.nets.build import get_model
from src.utils.utils import load_tango_3d_keypoints, load_camera_intrinsics, project_keypoints, quat2dcm
from src.utils.heatmaps import gaussian_heatmaps_from_keypoints, heatmap_to_keypoints_argmax


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


class _FullDataset(Dataset):
    def __init__(self, root, csv_path, num_keypoints=11, transforms=None):
        self.root = root
        self.num_keypoints = int(num_keypoints)
        self.transforms = transforms
        self.df = pd.read_csv(csv_path, header=None).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_numpy()
        imgpath_rel = str(row[0])
        imgpath = osp.join(self.root, imgpath_rel)
        img = Image.open(imgpath).convert('RGB')
        bbox = np.array(row[1:5], dtype=np.float32)
        q_gt = np.array(row[5:9], dtype=np.float32)
        t_gt = np.array(row[9:12], dtype=np.float32)
        keypts_pix = np.array(row[12:], dtype=np.float32).reshape(self.num_keypoints, 2).T

        img_t = img
        bbox_t = bbox
        keypts_norm = keypts_pix
        if self.transforms is not None:
            img_t, bbox_t, keypts_norm = self.transforms(img_t, bbox_t, keypts_norm)

        return {
            'idx': int(idx),
            'img_rel': imgpath_rel,
            'img_pil': img,
            'img_t': img_t,
            'bbox_orig': bbox,
            'bbox_crop': np.array(bbox_t, dtype=np.float32),
            'q_gt': q_gt,
            't_gt': t_gt,
            'keypts_pix': keypts_pix,
            'keypts_norm': np.array(keypts_norm, dtype=np.float32),
        }


def _camera_frame_xyz(q, t, pts3d):
    q = np.array(q, dtype=np.float64).reshape(4)
    t = np.array(t, dtype=np.float64).reshape(3)
    pts = np.array(pts3d, dtype=np.float64).reshape(-1, 3)
    R = quat2dcm(q)
    xyz = (R.T @ pts.T).T + t.reshape(1, 3)
    return xyz


def _argmax_xy(hm_2d):
    hm = hm_2d.reshape(-1)
    i = int(hm.argmax().item())
    w = int(hm_2d.shape[-1])
    y = i // w
    x = i - y * w
    return float(x), float(y)


def _topk_peaks(hm_2d, k=3):
    hm = hm_2d.reshape(-1)
    kk = int(k)
    kk = max(1, min(kk, int(hm.shape[0])))
    vals, idx = torch.topk(hm, k=kk, largest=True, sorted=True)
    w = int(hm_2d.shape[-1])
    out = []
    for j in range(kk):
        i = int(idx[j].item())
        y = i // w
        x = i - y * w
        out.append((float(x), float(y), float(vals[j].item())))
    return out


def _foreground_at_points(img_t, kp_norm, fg_thr=0.12, patch=3):
    img = img_t.detach().float()
    if img.ndim != 3:
        raise ValueError('img_t must be CHW tensor')
    C, H, W = img.shape
    gray = img.mean(dim=0)
    xs = np.array(kp_norm[0], dtype=np.float64).reshape(-1)
    ys = np.array(kp_norm[1], dtype=np.float64).reshape(-1)
    K = int(xs.shape[0])
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


def _foreground_scores(img_t, kp_norm, patch=3):
    img = img_t.detach().float()
    if img.ndim != 3:
        raise ValueError('img_t must be CHW tensor')
    C, H, W = img.shape
    gray = img.mean(dim=0)
    xs = np.array(kp_norm[0], dtype=np.float64).reshape(-1)
    ys = np.array(kp_norm[1], dtype=np.float64).reshape(-1)
    K = int(xs.shape[0])
    out = np.full((K,), np.nan, dtype=np.float64)
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
        out[k] = float(gray[y0:y1, x0:x1].mean().detach().cpu())
    return out


def main():
    ap = argparse.ArgumentParser('Audit worst keypoints (geometry + coord transforms)')
    ap.add_argument('--dataroot', type=str, required=True)
    ap.add_argument('--dataname', type=str, default='')
    ap.add_argument('--domain', type=str, default='synthetic')
    ap.add_argument('--csv', type=str, default='splits_krn/train.csv')
    ap.add_argument('--savedir', type=str, required=True)
    ap.add_argument('--indices_file', type=str, default='')
    ap.add_argument('--checkpoint', type=str, default='')
    ap.add_argument('--num_keypoints', type=int, default=11)
    ap.add_argument('--input_shape', nargs='+', type=int, default=(224, 224))
    ap.add_argument('--heatmap_size', nargs='+', type=int, default=(56, 56))
    ap.add_argument('--heatmap_sigma', type=float, default=2.0)
    ap.add_argument('--z_min', type=float, default=0.01)
    ap.add_argument('--fg_thr', type=float, default=0.12)
    ap.add_argument('--fg_patch', type=int, default=3)
    ap.add_argument('--idxs', type=str, default='')
    ap.add_argument('--use_worst5_from_best', action='store_true', default=True)
    args = ap.parse_args()

    root, csv_path = _resolve_root_and_csv(args.dataroot, args.dataname, args.domain, args.csv)
    os.makedirs(args.savedir, exist_ok=True)
    out_txt = osp.join(args.savedir, 'worst_keypoints_audit.txt')

    def _find_indices_file(savedir, explicit):
        if explicit.strip() != '':
            p = osp.abspath(explicit)
            if not osp.exists(p):
                raise FileNotFoundError(f'indices_file not found: {p}')
            return p
        cand = []
        sd = osp.abspath(savedir)
        cand.append(osp.join(sd, 'overfit_indices.txt'))
        cand.append(osp.join(osp.dirname(sd), 'overfit_indices.txt'))
        cand.append(osp.join(osp.dirname(osp.dirname(sd)), 'overfit_indices.txt'))
        for p in cand:
            if osp.exists(p):
                return p
        raise FileNotFoundError(
            'overfit_indices.txt not found. Tried:\n' + '\n'.join(cand) +
            '\nTip: pass --savedir to the overfit root folder that contains overfit_indices.txt, '
            'or pass --indices_file explicitly.'
        )

    idxs_path = _find_indices_file(args.savedir, args.indices_file)
    with open(idxs_path, 'r', encoding='utf-8') as f:
        idxs_line = f.readline().strip()
    subset_src = [int(x) for x in idxs_line.split(',') if x.strip() != '']

    chosen = None
    if args.idxs.strip() != '':
        chosen = [int(x) for x in args.idxs.split(',') if x.strip() != '']
    elif bool(args.use_worst5_from_best):
        best_path = osp.join(args.savedir, 'overfit_best.txt')
        if osp.exists(best_path):
            with open(best_path, 'r', encoding='utf-8') as f:
                txt = f.read()
            key = 'rmse_worst5='
            if key in txt:
                tail = txt.split(key, 1)[1].splitlines()[0].strip()
                worst = eval(tail, {'__builtins__': {}})
                chosen = [int(p[1]) for p in worst]
    if chosen is None:
        chosen = list(range(min(5, len(subset_src))))

    tfm = build_transforms('krn', (int(args.input_shape[0]), int(args.input_shape[1])), p_aug=0.0, is_train=False)
    ds = _FullDataset(root, csv_path, num_keypoints=int(args.num_keypoints), transforms=tfm)

    kp3d_path = osp.join(osp.dirname(__file__), 'src/utils/tangoPoints.mat')
    cam_path = osp.join(root, 'camera.json')
    corners3D = load_tango_3d_keypoints(kp3d_path)
    cameraMatrix, distCoeffs = load_camera_intrinsics(cam_path)

    class _Cfg: pass
    cfg = _Cfg()
    cfg.model_name = 'krn'
    cfg.dann = False
    cfg.krn_head = 'heatmap'
    cfg.num_keypoints = int(args.num_keypoints)
    cfg.num_classes = 5000
    cfg.num_neighbors = 5
    cfg.input_shape = (int(args.input_shape[0]), int(args.input_shape[1]))
    cfg.backbone = 'swin_tiny_patch4_window7_224'
    cfg.backbone_pretrained = False
    cfg.backbone_pretrained_path = ''
    cfg.input_normalize = 'imagenet'
    cfg.heatmap_size = (int(args.heatmap_size[0]), int(args.heatmap_size[1]))
    cfg.heatmap_sigma = float(args.heatmap_sigma)
    cfg.lr_backbone = 1e-5
    cfg.lr_head = 1e-4
    cfg.weight_decay = 0.0
    cfg.grad_clip_norm = 1.0
    cfg.use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0') if cfg.use_cuda else torch.device('cpu')

    model = get_model(cfg).to(device)
    ckpt = args.checkpoint.strip()
    if ckpt == '':
        ckpt = osp.join(args.savedir, 'checkpoint_overfit_best.pth')
    if not osp.exists(ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
    sd = torch.load(ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()

    z_min = float(args.z_min)
    Hm = (int(args.heatmap_size[0]), int(args.heatmap_size[1]))
    imgH = int(args.input_shape[0])
    imgW = int(args.input_shape[1])

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f'root={root}\n')
        f.write(f'csv={csv_path}\n')
        f.write(f'savedir={args.savedir}\n')
        f.write(f'checkpoint={ckpt}\n')
        f.write(f'input_shape={imgH}x{imgW}\n')
        f.write(f'heatmap_size={Hm[0]}x{Hm[1]}\n')
        f.write(f'z_min={z_min}\n')
        f.write(f'subset_src_indices={subset_src}\n')
        f.write(f'chosen_subset_indices={chosen}\n')

        for subset_i in chosen:
            if subset_i < 0 or subset_i >= len(subset_src):
                continue
            ds_i = subset_src[subset_i]
            ex = ds[ds_i]

            img_pil = ex['img_pil']
            org_w, org_h = img_pil.size
            bbox_crop = ex['bbox_crop'].reshape(-1)
            bb0, bb1, bb2, bb3 = float(bbox_crop[0]), float(bbox_crop[1]), float(bbox_crop[2]), float(bbox_crop[3])
            dx = max(float(bb1 - bb0), 1.0)
            dy = max(float(bb3 - bb2), 1.0)
            scx = max(float(imgW - 1), 1.0) / dx
            scy = max(float(imgH - 1), 1.0) / dy

            q_gt = ex['q_gt']
            t_gt = ex['t_gt']
            keypts_pix = ex['keypts_pix']
            keypts_norm = ex['keypts_norm']

            proj2d = project_keypoints(q_gt, t_gt, cameraMatrix, distCoeffs, corners3D.T)
            proj2d = np.array(proj2d, dtype=np.float64).reshape(2, -1)
            proj_err = np.linalg.norm(proj2d - keypts_pix.astype(np.float64), axis=0)

            xyz = _camera_frame_xyz(q_gt, t_gt, corners3D)
            u_orig = proj2d[0]
            v_orig = proj2d[1]
            u_crop = u_orig - bb0
            v_crop = v_orig - bb2
            u_img = u_crop / dx * max(float(imgW - 1), 1.0)
            v_img = v_crop / dy * max(float(imgH - 1), 1.0)
            u_hm = u_crop / dx * max(float(Hm[1] - 1), 1.0)
            v_hm = v_crop / dy * max(float(Hm[0] - 1), 1.0)

            gt_u_img = keypts_norm[0] * max(float(imgW - 1), 1.0)
            gt_v_img = keypts_norm[1] * max(float(imgH - 1), 1.0)
            gt_u_hm = keypts_norm[0] * max(float(Hm[1] - 1), 1.0)
            gt_v_hm = keypts_norm[1] * max(float(Hm[0] - 1), 1.0)

            valid_mask = np.isfinite(keypts_norm[0]) & np.isfinite(keypts_norm[1]) & (keypts_norm[0] >= 0) & (keypts_norm[0] <= 1) & (keypts_norm[1] >= 0) & (keypts_norm[1] <= 1)
            fg_val = _foreground_scores(ex['img_t'], keypts_norm, patch=int(args.fg_patch))
            is_fg = np.isfinite(fg_val) & (fg_val > float(args.fg_thr))
            in_orig = np.isfinite(u_orig) & np.isfinite(v_orig) & (u_orig >= 0) & (u_orig <= (org_w - 1)) & (v_orig >= 0) & (v_orig <= (org_h - 1))
            in_crop = np.isfinite(u_orig) & np.isfinite(v_orig) & (u_orig >= bb0) & (u_orig <= bb1) & (v_orig >= bb2) & (v_orig <= bb3)
            in_hm = np.isfinite(u_hm) & np.isfinite(v_hm) & (u_hm >= 0) & (u_hm <= (Hm[1] - 1)) & (v_hm >= 0) & (v_hm <= (Hm[0] - 1))
            zc = xyz[:, 2]
            z_ok = np.isfinite(zc) & (zc > z_min)
            geom_valid = z_ok & in_crop & in_hm & np.isfinite(u_orig) & np.isfinite(v_orig)
            fg_valid = is_fg
            geomfg_valid = geom_valid & fg_valid

            f.write('\n')
            f.write(f'=== sample subset_idx={subset_i} dataset_idx={ds_i} img={ex["img_rel"]} ===\n')
            f.write(f'orig_size={org_w}x{org_h} input_size={imgW}x{imgH} heatmap_size={Hm[1]}x{Hm[0]}\n')
            f.write(f'crop_box=[{bb0:.2f},{bb1:.2f},{bb2:.2f},{bb3:.2f}] scale_x={scx:.6f} scale_y={scy:.6f}\n')
            f.write(f'foreground_thr={float(args.fg_thr):.6f} foreground_patch={int(args.fg_patch)}\n')
            img_min = float(ex['img_t'].detach().min().cpu())
            img_max = float(ex['img_t'].detach().max().cpu())
            img_mean = float(ex['img_t'].detach().float().mean().cpu())
            f.write(f'img_t min/max/mean={img_min:.6f}/{img_max:.6f}/{img_mean:.6f}\n')
            f.write(f'proj_vs_label_err_px mean={float(np.mean(proj_err)):.3f} median={float(np.median(proj_err)):.3f} max={float(np.max(proj_err)):.3f}\n')

            img_t = ex['img_t'].unsqueeze(0).to(device)
            key_t = torch.as_tensor(keypts_norm, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(img_t, key_t)
                pred_hm = getattr(model, 'last_pred_heatmaps', None)
                if pred_hm is None:
                    f.write('model_last_pred_heatmaps=None\n')
                    continue
                pred_hm = pred_hm.squeeze(0).detach().cpu()
                pred_kp, _ = heatmap_to_keypoints_argmax(pred_hm.unsqueeze(0))
                pred_kp = pred_kp.squeeze(0).detach().cpu().numpy()

            gt_hm, valid_hm = gaussian_heatmaps_from_keypoints(torch.as_tensor(keypts_norm, dtype=torch.float32).unsqueeze(0), Hm, sigma=float(args.heatmap_sigma))
            gt_hm = gt_hm.squeeze(0).detach().cpu()
            gt_kp_dec, _ = heatmap_to_keypoints_argmax(gt_hm.unsqueeze(0))
            gt_kp_dec = gt_kp_dec.squeeze(0).detach().cpu().numpy()

            pred_u_img = pred_kp[0] * max(float(imgW - 1), 1.0)
            pred_v_img = pred_kp[1] * max(float(imgH - 1), 1.0)

            per_err_img = np.sqrt((pred_u_img - gt_u_img) ** 2 + (pred_v_img - gt_v_img) ** 2)

            for k in range(int(args.num_keypoints)):
                kp3 = corners3D[k]
                Xc, Yc, Zc = float(xyz[k, 0]), float(xyz[k, 1]), float(xyz[k, 2])
                uo, vo = float(u_orig[k]), float(v_orig[k])
                uc, vc = float(u_crop[k]), float(v_crop[k])
                ui, vi = float(u_img[k]), float(v_img[k])
                uh, vh = float(u_hm[k]), float(v_hm[k])
                vm = bool(valid_mask[k])
                io = bool(in_orig[k])
                ic = bool(in_crop[k])
                ih = bool(in_hm[k])
                gz = bool(z_ok[k])
                gv = bool(geom_valid[k])
                fg = bool(fg_valid[k])
                gfg = bool(geomfg_valid[k])
                fg_v = float(fg_val[k]) if np.isfinite(fg_val[k]) else float('nan')

                gt_peak_x, gt_peak_y = _argmax_xy(gt_hm[k])
                gt_dec_x = float(gt_kp_dec[0, k] * max(float(Hm[1] - 1), 1.0))
                gt_dec_y = float(gt_kp_dec[1, k] * max(float(Hm[0] - 1), 1.0))
                pr_peak_x, pr_peak_y = _argmax_xy(pred_hm[k])
                pr_ui = float(pred_u_img[k])
                pr_vi = float(pred_v_img[k])
                err_px = float(per_err_img[k])

                pr_top3 = _topk_peaks(pred_hm[k], k=3)
                gt_top3 = _topk_peaks(gt_hm[k], k=3)
                pr_ratio = float('nan')
                if len(pr_top3) >= 2:
                    pr_ratio = pr_top3[0][2] / max(pr_top3[1][2], 1e-12)
                pr_d1 = float(np.sqrt((pr_top3[0][0] - gt_peak_x) ** 2 + (pr_top3[0][1] - gt_peak_y) ** 2)) if len(pr_top3) >= 1 else float('nan')

                f.write(
                    f'k={k:02d} '
                    f'3D=({kp3[0]:.5f},{kp3[1]:.5f},{kp3[2]:.5f}) '
                    f'cam=({Xc:.5f},{Yc:.5f},{Zc:.5f}) '
                    f'z_ok={int(gz)} '
                    f'u_orig/v_orig=({uo:.2f},{vo:.2f}) '
                    f'u_crop/v_crop=({uc:.2f},{vc:.2f}) '
                    f'u_img/v_img=({ui:.2f},{vi:.2f}) '
                    f'u_hm/v_hm=({uh:.2f},{vh:.2f}) '
                    f'valid={int(vm)} in_orig={int(io)} in_crop={int(ic)} in_hm={int(ih)} '
                    f'fg_val={fg_v:.4f} is_fg={int(fg)} geom_valid={int(gv)} geomfg_valid={int(gfg)} '
                    f'gt_peak=({gt_peak_x:.2f},{gt_peak_y:.2f}) gt_dec=({gt_dec_x:.2f},{gt_dec_y:.2f}) '
                    f'pred_peak=({pr_peak_x:.2f},{pr_peak_y:.2f}) pred_img=({pr_ui:.2f},{pr_vi:.2f}) '
                    f'pred_ratio={pr_ratio:.3f} pred_d1={pr_d1:.3f} '
                    f'pred_top3={pr_top3} gt_top3={gt_top3} '
                    f'err_px={err_px:.3f}\n'
                )

    print(f'Wrote: {out_txt}')


if __name__ == '__main__':
    main()
