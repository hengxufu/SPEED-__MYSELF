import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.datasets.transforms import build_transforms
from src.utils.heatmaps import gaussian_heatmaps_from_keypoints


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
        return img, torch.as_tensor(bbox, dtype=torch.float32), torch.as_tensor(keypts, dtype=torch.float32)


def _overlay_and_save(savefn, image, keypts_norm, gt_hm, valid_mask):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = image.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
    _, h, w = image.shape
    sx = max(float(w - 1), 1.0)
    sy = max(float(h - 1), 1.0)

    kp = keypts_norm.detach().cpu()
    xs = kp[0].numpy() * sx
    ys = kp[1].numpy() * sy
    vm = valid_mask.detach().cpu().numpy().astype(bool).reshape(-1)

    hm = gt_hm.detach().cpu()
    hm_max = hm.max(dim=0).values
    hm_max = hm_max / max(float(hm_max.max().item()), 1e-12)

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(hm_max.numpy(), cmap='jet', alpha=0.45, vmin=0.0, vmax=1.0)
    plt.xlim(0, w - 1)
    plt.ylim(h - 1, 0)
    plt.scatter(xs[vm], ys[vm], c='lime', marker='+', label='gt')
    plt.axis('off')
    plt.legend(loc='lower right')
    os.makedirs(osp.dirname(savefn), exist_ok=True)
    plt.savefig(savefn, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser('Debug GT heatmap generation')
    ap.add_argument('--dataroot', type=str, required=True)
    ap.add_argument('--dataname', type=str, default='')
    ap.add_argument('--domain', type=str, default='synthetic')
    ap.add_argument('--csv', type=str, default='splits_krn/train.csv')
    ap.add_argument('--num_keypoints', type=int, default=11)
    ap.add_argument('--input_shape', nargs='+', type=int, default=(224, 224))
    ap.add_argument('--heatmap_size', nargs='+', type=int, default=(56, 56))
    ap.add_argument('--heatmap_sigma', type=float, default=2.0)
    ap.add_argument('--num', type=int, default=16)
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--savedir', type=str, default='log/debug_gt_heatmap')
    args = ap.parse_args()

    rng = np.random.RandomState(int(args.seed))
    root, csv_path = _resolve_root_and_csv(args.dataroot, args.dataname, args.domain, args.csv)

    tfm = build_transforms('krn', (int(args.input_shape[0]), int(args.input_shape[1])), p_aug=0.0, is_train=False)
    ds = _Dataset(root, csv_path, num_keypoints=int(args.num_keypoints), transforms=tfm)

    idxs = rng.choice(len(ds), size=min(int(args.num), len(ds)), replace=False).tolist()
    sub = torch.utils.data.Subset(ds, idxs)
    dl = DataLoader(sub, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    Hm = (int(args.heatmap_size[0]), int(args.heatmap_size[1]))
    sigma = float(args.heatmap_sigma)

    all_max = []
    for i, (img, bbox, keypts) in enumerate(dl):
        img = img.squeeze(0)
        keypts = keypts.squeeze(0)
        gt_hm, valid = gaussian_heatmaps_from_keypoints(keypts.unsqueeze(0), Hm, sigma=sigma)
        gt_hm = gt_hm.squeeze(0)
        valid = valid.squeeze(0)

        vm = valid.detach().cpu().numpy().astype(bool).reshape(-1)
        hm_np = gt_hm.detach().cpu().numpy()
        per_k_max = hm_np.reshape(hm_np.shape[0], -1).max(axis=1)

        print(f'--- sample {i:02d} ---')
        print('valid_kpts=', int(vm.sum()), '/', int(valid.numel()))
        print('gt_hm min/max/mean/std=', float(hm_np.min()), float(hm_np.max()), float(hm_np.mean()), float(hm_np.std()))
        if int(vm.sum()) > 0:
            print('per_kpt_max(valid)=', per_k_max[vm].tolist())
            all_max.extend(per_k_max[vm].tolist())

        _overlay_and_save(
            osp.join(args.savedir, f'sample_{i:02d}.png'),
            img.detach().cpu(),
            keypts.detach().cpu(),
            gt_hm,
            valid,
        )

    if len(all_max) > 0:
        all_max = np.array(all_max, dtype=np.float32)
        print('all_valid_keypoints max(mean/min/median)=', float(all_max.mean()), float(all_max.min()), float(np.median(all_max)))
    print('savedir=', osp.abspath(args.savedir))


if __name__ == '__main__':
    main()
