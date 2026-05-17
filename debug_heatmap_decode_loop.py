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


class _Dataset(Dataset):
    def __init__(self, root, csv_path, num_keypoints=11, transforms=None):
        self.root = root
        self.num_keypoints = int(num_keypoints)
        self.transforms = transforms
        self.df = pd.read_csv(csv_path, header=None).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_numpy()
        imgpath = osp.join(self.root, row[0])
        img = Image.open(imgpath).convert('RGB')
        bbox = np.array(row[1:5], dtype=np.float32)
        keypts = np.array(row[12:], dtype=np.float32).reshape(self.num_keypoints, 2).T
        if self.transforms is not None:
            img, bbox, keypts = self.transforms(img, bbox, keypts)
        return img, torch.as_tensor(keypts, dtype=torch.float32)


def _save_vis(savefn, image, kp_gt, kp_dec, rmse_px):
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = image.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
    _, h, w = image.shape
    sx = max(float(w - 1), 1.0)
    sy = max(float(h - 1), 1.0)
    xs_gt = kp_gt[0].cpu().numpy() * sx
    ys_gt = kp_gt[1].cpu().numpy() * sy
    xs_dc = kp_dec[0].cpu().numpy() * sx
    ys_dc = kp_dec[1].cpu().numpy() * sy

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.xlim(0, w - 1)
    plt.ylim(h - 1, 0)
    plt.scatter(xs_gt, ys_gt, c='lime', marker='+', label='gt')
    plt.scatter(xs_dc, ys_dc, c='red', marker='x', label='decoded')
    plt.title(f'rmse_px={rmse_px:.3f}')
    plt.axis('off')
    plt.legend(loc='lower right')
    os.makedirs(osp.dirname(savefn), exist_ok=True)
    plt.savefig(savefn, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser('Debug GT->heatmap->decode loop')
    ap.add_argument('--dataroot', type=str, required=True)
    ap.add_argument('--dataname', type=str, default='')
    ap.add_argument('--domain', type=str, default='synthetic')
    ap.add_argument('--csv', type=str, default='splits_krn/train.csv')
    ap.add_argument('--num_keypoints', type=int, default=11)
    ap.add_argument('--input_shape', nargs='+', type=int, default=(224, 224))
    ap.add_argument('--heatmap_size', nargs='+', type=int, default=(56, 56))
    ap.add_argument('--heatmap_sigma', type=float, default=2.0)
    ap.add_argument('--decode', type=str, default='argmax', choices=['argmax', 'softargmax'])
    ap.add_argument('--beta', type=float, default=100.0)
    ap.add_argument('--num', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--savedir', type=str, default='log/debug_heatmap_decode_loop')
    ap.add_argument('--save_n', type=int, default=16)
    args = ap.parse_args()

    root, csv_path = _resolve_root_and_csv(args.dataroot, args.dataname, args.domain, args.csv)
    tfm = build_transforms('krn', (int(args.input_shape[0]), int(args.input_shape[1])), p_aug=0.0, is_train=False)
    ds = _Dataset(root, csv_path, num_keypoints=int(args.num_keypoints), transforms=tfm)
    ds = torch.utils.data.Subset(ds, list(range(min(int(args.num), len(ds)))))
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)

    Hm = (int(args.heatmap_size[0]), int(args.heatmap_size[1]))
    sigma = float(args.heatmap_sigma)
    img_size = (int(args.input_shape[0]), int(args.input_shape[1]))

    saved = 0
    rmses = []
    for images, keypts in dl:
        gt_hm, valid = gaussian_heatmaps_from_keypoints(keypts, Hm, sigma=sigma)
        if args.decode == 'softargmax':
            dec_kp, _ = heatmap_to_keypoints_softargmax(gt_hm, beta=float(args.beta))
        else:
            dec_kp, _ = heatmap_to_keypoints_argmax(gt_hm)

        rmse = keypoint_rmse_pixels(dec_kp, keypts, image_size=img_size, valid_mask=valid)
        inside = inside_image_percentage(dec_kp, valid)
        rmses.append(float(rmse.detach().cpu()))
        print('batch rmse_px=', float(rmse.detach().cpu()), 'inside_pct=', float(inside.detach().cpu()))

        per_err = per_keypoint_error_pixels(dec_kp, keypts, image_size=img_size).detach().cpu().numpy()
        for b in range(images.shape[0]):
            if saved < int(args.save_n):
                _save_vis(
                    osp.join(args.savedir, f'sample_{saved:03d}_rmse{rmses[-1]:.3f}.png'),
                    images[b].detach().cpu(),
                    keypts[b].detach().cpu(),
                    dec_kp[b].detach().cpu(),
                    rmses[-1],
                )
                saved += 1
        if len(rmses) >= max(1, int(args.num) // max(int(args.batch_size), 1)):
            pass

    print('rmse_px mean/median=', float(np.mean(rmses)) if len(rmses) else float('nan'), float(np.median(rmses)) if len(rmses) else float('nan'))
    print('savedir=', osp.abspath(args.savedir))


if __name__ == '__main__':
    main()
