from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.Park2019KRNDataset import Park2019KRNDataset  # noqa: E402
from src.datasets.transforms import build_transforms  # noqa: E402
from src.nets.build import get_model  # noqa: E402
from src.utils.heatmap_pipeline import keypoints_to_pose_ransac, reprojection_errors_px  # noqa: E402
from src.utils.utils import load_camera_intrinsics, load_tango_3d_keypoints, project_keypoints, set_all_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Geometry-filtered target pre-adaptation for heatmap KRN")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--domain", required=True, choices=["lightbox", "sunlamp"])
    p.add_argument("--test_csv", default="test.csv")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_adapt_samples", type=int, default=0)
    p.add_argument("--min_inliers", type=int, default=6)
    p.add_argument("--reproj_thr_px", type=float, default=8.0)
    p.add_argument("--reproj_median_thr_px", type=float, default=8.0)
    p.add_argument("--pose_t_min", type=float, default=0.5)
    p.add_argument("--pose_t_max", type=float, default=20.0)
    p.add_argument("--adapt_backbone_norm", action="store_true")
    p.add_argument("--refresh_teacher_each_epoch", action="store_true")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def make_cfg(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        projroot=str(REPO_ROOT),
        dataroot=str(Path(args.dataroot).resolve()),
        dataname="",
        model_name="krn",
        dann=False,
        krn_head="heatmap",
        input_shape=(224, 224),
        num_keypoints=11,
        backbone="swin_tiny_patch4_window7_224",
        backbone_pretrained=False,
        backbone_pretrained_path=str(REPO_ROOT / "checkpoints" / "pretrained" / "swin_tiny_patch4_window7_224.pth"),
        debug_shapes=False,
        backbone_fpn=True,
        backbone_out_indices=[1, 2, 3],
        backbone_out_index=2,
        input_normalize="imagenet",
        heatmap_size=(56, 56),
        heatmap_sigma=2.0,
        heatmap_decode="softargmax",
        heatmap_beta=100.0,
        heatmap_activation="none",
        heatmap_loss="heatmap_ce_coord_aux",
        heatmap_pos_thr=0.1,
        heatmap_neg_weight=0.01,
        coord_aux_weight=0.2,
        heatmap_hard_kpts="3,4,5,6,10",
        heatmap_hard_kpt_weight=1.5,
        heatmap_other_kpt_weight=1.0,
        return_test_keypts=True,
        deterministic_crop=True,
        p_aug=0.0,
        test_domain=args.domain,
        test_csv=args.test_csv,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        keypts_3d_model=str(REPO_ROOT / "src" / "utils" / "tangoPoints.mat"),
        attitude_class=str(REPO_ROOT / "src" / "utils" / "attitudeClasses.mat"),
    )


def load_weights(model: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)


def trainable_target_params(model: torch.nn.Module, adapt_backbone_norm: bool = False) -> list[torch.nn.Parameter]:
    prefixes = ("fpn_lateral.", "fpn_out.", "neck.", "decoder.", "head.")
    trainable = []
    for name, param in model.named_parameters():
        lower_name = name.lower()
        is_head_param = name.startswith(prefixes)
        is_backbone_norm = (
            adapt_backbone_norm
            and name.startswith("backbone.")
            and ("norm" in lower_name or ".bn" in lower_name or "batchnorm" in lower_name)
        )
        param.requires_grad = bool(is_head_param or is_backbone_norm)
        if param.requires_grad:
            trainable.append(param)
    return trainable


def make_loader(cfg: SimpleNamespace, max_samples: int, batch_size: int, num_workers: int) -> DataLoader:
    transforms = build_transforms(
        cfg.model_name,
        cfg.input_shape,
        p_aug=0.0,
        is_train=False,
        deterministic_crop=True,
    )
    dataset = Park2019KRNDataset(cfg, transforms, is_train=False, is_source=False, load_labels=True)
    if max_samples and max_samples > 0:
        n = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)


def build_pseudo_batch(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    bbox: torch.Tensor,
    corners3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = int(pred_x.shape[0])
    num_keypoints = int(pred_x.shape[1])
    pseudo = torch.full((batch_size, 2, num_keypoints), float("nan"), dtype=torch.float32)
    accepted = 0
    inliers_all: list[int] = []
    reproj_meds: list[float] = []

    for b in range(batch_size):
        kp = torch.stack([pred_x[b].detach().cpu(), pred_y[b].detach().cpu()], dim=0)
        bb = bbox[b].detach().cpu().numpy().astype(np.float64).reshape(4)
        try:
            q_r, t_r, inlier_idx, _ = keypoints_to_pose_ransac(
                kp,
                bb,
                corners3d,
                camera_matrix,
                dist_coeffs,
                valid_mask=None,
                reproj_thr_px=args.reproj_thr_px,
            )
            inlier_idx = np.asarray(inlier_idx, dtype=np.int64).reshape(-1)
            if int(inlier_idx.size) < int(args.min_inliers):
                continue
            t_norm = float(np.linalg.norm(np.asarray(t_r, dtype=np.float64).reshape(3)))
            if (not math.isfinite(t_norm)) or t_norm < args.pose_t_min or t_norm > args.pose_t_max:
                continue
            err = reprojection_errors_px(kp, bb, q_r, t_r, corners3d, camera_matrix, dist_coeffs)
            inlier_err = np.asarray(err, dtype=np.float64)[inlier_idx]
            med = float(np.median(inlier_err[np.isfinite(inlier_err)])) if np.isfinite(inlier_err).any() else float("inf")
            if (not math.isfinite(med)) or med > args.reproj_median_thr_px:
                continue

            proj = project_keypoints(
                np.asarray(q_r, dtype=np.float64),
                np.asarray(t_r, dtype=np.float64),
                camera_matrix,
                dist_coeffs,
                np.asarray(corners3d, dtype=np.float64),
            )
            dx = max(float(bb[1] - bb[0]), 1.0)
            dy = max(float(bb[3] - bb[2]), 1.0)
            x = (np.asarray(proj[0], dtype=np.float64).reshape(-1) - float(bb[0])) / dx
            y = (np.asarray(proj[1], dtype=np.float64).reshape(-1) - float(bb[2])) / dy
            valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
            if int(valid.sum()) < int(args.min_inliers):
                continue

            pseudo[b, 0, valid] = torch.from_numpy(x[valid].astype(np.float32))
            pseudo[b, 1, valid] = torch.from_numpy(y[valid].astype(np.float32))
            accepted += 1
            inliers_all.append(int(inlier_idx.size))
            reproj_meds.append(float(med))
        except Exception:
            continue

    stats = {
        "accepted": float(accepted),
        "batch": float(batch_size),
        "accepted_pct": float(100.0 * accepted / max(batch_size, 1)),
        "inliers_mean": float(np.mean(inliers_all)) if inliers_all else 0.0,
        "reproj_median_px": float(np.median(reproj_meds)) if reproj_meds else float("nan"),
    }
    return pseudo, stats


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    cfg = make_cfg(args)
    device = torch.device("cuda:0") if (torch.cuda.is_available() and not args.no_cuda) else torch.device("cpu")
    set_all_seeds(args.seed, cfg, device.type == "cuda")

    model = get_model(cfg)
    load_weights(model, args.checkpoint, device)
    teacher = get_model(cfg)
    load_weights(teacher, args.checkpoint, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    trainable = trainable_target_params(model, adapt_backbone_norm=args.adapt_backbone_norm)
    logging.info("Trainable target parameters: %d / %d", sum(p.numel() for p in trainable), sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    loader = make_loader(cfg, args.max_adapt_samples, args.batch_size, args.num_workers)
    camera_matrix, dist_coeffs = load_camera_intrinsics(str(Path(cfg.dataroot) / "camera.json"))
    corners3d = load_tango_3d_keypoints(cfg.keypts_3d_model)

    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        teacher.eval()
        total_loss = 0.0
        total_steps = 0
        total_seen = 0
        total_accepted = 0
        reproj_meds = []
        inlier_means = []

        for step, batch in enumerate(loader, start=1):
            images, bbox, _keypts_gt, _q_gt, _t_gt = batch
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                pred_x, pred_y = teacher(images)
            pseudo, pseudo_stats = build_pseudo_batch(
                pred_x,
                pred_y,
                bbox,
                corners3d,
                camera_matrix,
                dist_coeffs,
                args,
            )
            total_seen += int(pseudo_stats["batch"])
            total_accepted += int(pseudo_stats["accepted"])
            if math.isfinite(float(pseudo_stats["reproj_median_px"])):
                reproj_meds.append(float(pseudo_stats["reproj_median_px"]))
            if pseudo_stats["inliers_mean"] > 0:
                inlier_means.append(float(pseudo_stats["inliers_mean"]))

            if int(pseudo_stats["accepted"]) <= 0:
                continue

            pseudo = pseudo.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, summary = model(images, pseudo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_steps += 1

            if step == 1 or step % 50 == 0 or step == len(loader):
                logging.info(
                    "epoch=%d step=%d/%d loss=%.4f accepted=%d/%d total_accept=%.1f%%",
                    epoch + 1,
                    step,
                    len(loader),
                    float(loss.detach().cpu()),
                    int(pseudo_stats["accepted"]),
                    int(pseudo_stats["batch"]),
                    100.0 * total_accepted / max(total_seen, 1),
                )

        row = {
            "epoch": epoch + 1,
            "loss_mean": total_loss / max(total_steps, 1),
            "optim_steps": total_steps,
            "seen": total_seen,
            "accepted": total_accepted,
            "accepted_pct": 100.0 * total_accepted / max(total_seen, 1),
            "pseudo_reproj_median_px": float(np.median(reproj_meds)) if reproj_meds else float("nan"),
            "pseudo_inliers_mean": float(np.mean(inlier_means)) if inlier_means else 0.0,
        }
        history.append(row)
        logging.info("epoch summary: %s", row)
        if args.refresh_teacher_each_epoch and epoch + 1 < int(args.epochs):
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            logging.info("teacher refreshed from student after epoch %d", epoch + 1)

    torch.save(model.state_dict(), outdir / "model_adapted.pth.tar")
    (outdir / "adapt_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "adapt_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("saved adapted model to %s", outdir / "model_adapted.pth.tar")


if __name__ == "__main__":
    main()
