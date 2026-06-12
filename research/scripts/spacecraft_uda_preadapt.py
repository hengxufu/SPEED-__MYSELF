from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.Park2019KRNDataset import Park2019KRNDataset  # noqa: E402
from src.datasets.transforms import build_transforms  # noqa: E402
from src.nets.build import get_model  # noqa: E402
from src.utils.heatmap_pipeline import keypoints_to_pose_ransac, reprojection_errors_px  # noqa: E402
from src.utils.utils import load_camera_intrinsics, load_tango_3d_keypoints, project_keypoints, set_all_seeds  # noqa: E402


@dataclass
class PseudoLabel:
    index: int
    q_wxyz: list[float]
    t_xyz_m: list[float]
    keypoints_norm: list[list[float]]
    inliers_a: int
    inliers_b: int
    reproj_a_px: float
    reproj_b_px: float
    pose_angle_deg: float
    translation_rel: float
    keypoint_disagreement_px: float


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset[index]


class PseudoTargetDataset(Dataset):
    def __init__(self, target_dataset: Dataset, pseudo_labels: list[PseudoLabel]):
        self.target_dataset = target_dataset
        self.pseudo_labels = pseudo_labels

    def __len__(self) -> int:
        return len(self.pseudo_labels)

    def __getitem__(self, index: int):
        pseudo = self.pseudo_labels[index]
        image, _bbox, _kp_gt, _q_gt, _t_gt = self.target_dataset[pseudo.index]
        keypoints = torch.tensor(pseudo.keypoints_norm, dtype=torch.float32)
        return image, keypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-round target pre-adaptation inspired by JotaBravo/spacecraft-uda. "
            "Pseudo-labels require cross-view teacher consensus and RANSAC-PnP validity."
        )
    )
    parser.add_argument("--dataroot", required=True, help="Prepared target root containing camera.json")
    parser.add_argument("--source_dataroot", default="", help="Optional SPEED+ root for synthetic replay")
    parser.add_argument("--domain", required=True, choices=["lightbox", "sunlamp"])
    parser.add_argument("--test_csv", default="test.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_b_checkpoint", default="")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs_per_round", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--final_lr_scale", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_adapt_samples", type=int, default=0)
    parser.add_argument("--min_inliers", type=int, default=6)
    parser.add_argument("--ransac_reproj_thr_px", type=float, default=8.0)
    parser.add_argument("--reproj_median_thr_px", type=float, default=8.0)
    parser.add_argument("--pose_t_min", type=float, default=0.5)
    parser.add_argument("--pose_t_max", type=float, default=20.0)
    parser.add_argument("--consensus_angle_deg", type=float, default=15.0)
    parser.add_argument("--consensus_translation_rel", type=float, default=0.20)
    parser.add_argument("--consensus_keypoint_px", type=float, default=15.0)
    parser.add_argument("--view_contrast", type=float, default=0.15)
    parser.add_argument("--view_brightness", type=float, default=0.06)
    parser.add_argument("--view_noise_std", type=float, default=0.01)
    parser.add_argument("--source_replay_weight", type=float, default=0.20)
    parser.add_argument("--source_batch_size", type=int, default=16)
    parser.add_argument("--adapt_backbone_norm", action="store_true")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def make_cfg(
    args: argparse.Namespace,
    dataroot: str,
    *,
    test_domain: str | None = None,
    test_csv: str = "test.csv",
) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        projroot=str(REPO_ROOT),
        dataroot=str(Path(dataroot).expanduser().resolve()),
        dataname="",
        model_name="krn",
        dann=False,
        krn_head="heatmap",
        input_shape=(224, 224),
        num_keypoints=11,
        backbone="swin_tiny_patch4_window7_224",
        backbone_pretrained=False,
        backbone_pretrained_path="",
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
        train_domain="synthetic",
        train_csv="train.csv",
        test_domain=test_domain or args.domain,
        test_csv=test_csv,
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


def trainable_target_params(model: torch.nn.Module, adapt_backbone_norm: bool) -> list[torch.nn.Parameter]:
    prefixes = ("fpn_lateral.", "fpn_out.", "neck.", "decoder.", "head.")
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        lower_name = name.lower()
        is_head = name.startswith(prefixes)
        is_backbone_norm = (
            adapt_backbone_norm
            and name.startswith("backbone.")
            and ("norm" in lower_name or ".bn" in lower_name or "batchnorm" in lower_name)
        )
        parameter.requires_grad = bool(is_head or is_backbone_norm)
        if parameter.requires_grad:
            trainable.append(parameter)
    return trainable


def make_target_dataset(cfg: SimpleNamespace, max_samples: int) -> Dataset:
    transforms = build_transforms(
        cfg.model_name,
        cfg.input_shape,
        p_aug=0.0,
        is_train=False,
        deterministic_crop=True,
    )
    dataset: Dataset = Park2019KRNDataset(cfg, transforms, is_train=False, is_source=False, load_labels=True)
    if max_samples > 0:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def make_source_loader(args: argparse.Namespace) -> DataLoader | None:
    if args.source_replay_weight <= 0 or not args.source_dataroot:
        return None
    cfg = make_cfg(args, args.source_dataroot, test_domain="synthetic")
    transforms = build_transforms(
        cfg.model_name,
        cfg.input_shape,
        p_aug=0.0,
        is_train=True,
        deterministic_crop=True,
    )
    dataset = Park2019KRNDataset(cfg, transforms, is_train=True, is_source=True, load_labels=True)
    return DataLoader(
        dataset,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def photometric_view(images: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    batch = images.shape[0]
    contrast = 1.0 + (torch.rand(batch, 1, 1, 1, device=images.device) * 2.0 - 1.0) * args.view_contrast
    brightness = (torch.rand(batch, 1, 1, 1, device=images.device) * 2.0 - 1.0) * args.view_brightness
    noise = torch.randn_like(images) * args.view_noise_std
    return torch.clamp(images * contrast + brightness + noise, 0.0, 1.0)


def quaternion_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    qa = np.asarray(q_a, dtype=np.float64).reshape(4)
    qb = np.asarray(q_b, dtype=np.float64).reshape(4)
    qa /= max(float(np.linalg.norm(qa)), 1e-12)
    qb /= max(float(np.linalg.norm(qb)), 1e-12)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def average_quaternion(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    qa = np.asarray(q_a, dtype=np.float64).reshape(4)
    qb = np.asarray(q_b, dtype=np.float64).reshape(4)
    qa /= max(float(np.linalg.norm(qa)), 1e-12)
    qb /= max(float(np.linalg.norm(qb)), 1e-12)
    if float(np.dot(qa, qb)) < 0:
        qb = -qb
    q = qa + qb
    return q / max(float(np.linalg.norm(q)), 1e-12)


def keypoint_disagreement_px(kp_a: torch.Tensor, kp_b: torch.Tensor, bbox: np.ndarray) -> float:
    bb = np.asarray(bbox, dtype=np.float64).reshape(4)
    scale = torch.tensor(
        [max(float(bb[1] - bb[0]), 1.0), max(float(bb[3] - bb[2]), 1.0)],
        dtype=torch.float32,
    ).view(2, 1)
    distance = torch.sqrt(torch.sum(((kp_a.cpu() - kp_b.cpu()) * scale) ** 2, dim=0))
    return float(distance.mean())


def pose_candidate(
    kp: torch.Tensor,
    bbox: np.ndarray,
    corners3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
) -> dict | None:
    try:
        q, t, inliers, _ = keypoints_to_pose_ransac(
            kp,
            bbox,
            corners3d,
            camera_matrix,
            dist_coeffs,
            valid_mask=None,
            reproj_thr_px=args.ransac_reproj_thr_px,
        )
        inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
        if int(inliers.size) < args.min_inliers:
            return None
        t_norm = float(np.linalg.norm(np.asarray(t, dtype=np.float64).reshape(3)))
        if not math.isfinite(t_norm) or t_norm < args.pose_t_min or t_norm > args.pose_t_max:
            return None
        errors = reprojection_errors_px(kp, bbox, q, t, corners3d, camera_matrix, dist_coeffs)
        inlier_errors = np.asarray(errors, dtype=np.float64)[inliers]
        reproj = float(np.median(inlier_errors[np.isfinite(inlier_errors)]))
        if not math.isfinite(reproj) or reproj > args.reproj_median_thr_px:
            return None
        return {"q": np.asarray(q), "t": np.asarray(t), "inliers": int(inliers.size), "reproj": reproj}
    except Exception:
        return None


def fused_pseudo_keypoints(
    q: np.ndarray,
    t: np.ndarray,
    bbox: np.ndarray,
    corners3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_valid: int,
) -> torch.Tensor | None:
    bb = np.asarray(bbox, dtype=np.float64).reshape(4)
    projected = project_keypoints(q, t, camera_matrix, dist_coeffs, corners3d)
    dx = max(float(bb[1] - bb[0]), 1.0)
    dy = max(float(bb[3] - bb[2]), 1.0)
    x = (np.asarray(projected[0], dtype=np.float64).reshape(-1) - float(bb[0])) / dx
    y = (np.asarray(projected[1], dtype=np.float64).reshape(-1) - float(bb[2])) / dy
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
    if int(valid.sum()) < min_valid:
        return None
    pseudo = torch.full((2, len(x)), float("nan"), dtype=torch.float32)
    pseudo[0, valid] = torch.from_numpy(x[valid].astype(np.float32))
    pseudo[1, valid] = torch.from_numpy(y[valid].astype(np.float32))
    return pseudo


@torch.no_grad()
def generate_pseudo_labels(
    teacher_a: torch.nn.Module,
    teacher_b: torch.nn.Module,
    target_dataset: Dataset,
    corners3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[PseudoLabel], dict[str, float]]:
    loader = DataLoader(
        IndexedDataset(target_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    teacher_a.eval()
    teacher_b.eval()
    accepted: list[PseudoLabel] = []
    geometry_pass_a = 0
    geometry_pass_b = 0
    consensus_reject = 0

    for step, (indices, batch) in enumerate(loader, start=1):
        images, bbox, _kp_gt, _q_gt, _t_gt = batch
        images = images.to(device, non_blocking=True)
        view_b = photometric_view(images, args)
        pred_ax, pred_ay = teacher_a(images)
        pred_bx, pred_by = teacher_b(view_b)

        for local_index, dataset_index in enumerate(indices.tolist()):
            kp_a = torch.stack([pred_ax[local_index], pred_ay[local_index]], dim=0)
            kp_b = torch.stack([pred_bx[local_index], pred_by[local_index]], dim=0)
            bb = bbox[local_index].cpu().numpy().astype(np.float64).reshape(4)
            pose_a = pose_candidate(kp_a, bb, corners3d, camera_matrix, dist_coeffs, args)
            pose_b = pose_candidate(kp_b, bb, corners3d, camera_matrix, dist_coeffs, args)
            geometry_pass_a += int(pose_a is not None)
            geometry_pass_b += int(pose_b is not None)
            if pose_a is None or pose_b is None:
                continue

            angle = quaternion_angle_deg(pose_a["q"], pose_b["q"])
            t_a = np.asarray(pose_a["t"], dtype=np.float64).reshape(3)
            t_b = np.asarray(pose_b["t"], dtype=np.float64).reshape(3)
            t_rel = float(np.linalg.norm(t_a - t_b) / max(np.linalg.norm(t_a), np.linalg.norm(t_b), 1e-6))
            kp_disagreement = keypoint_disagreement_px(kp_a, kp_b, bb)
            if (
                angle > args.consensus_angle_deg
                or t_rel > args.consensus_translation_rel
                or kp_disagreement > args.consensus_keypoint_px
            ):
                consensus_reject += 1
                continue

            q_fused = average_quaternion(pose_a["q"], pose_b["q"])
            t_fused = 0.5 * (t_a + t_b)
            pseudo = fused_pseudo_keypoints(
                q_fused,
                t_fused,
                bb,
                corners3d,
                camera_matrix,
                dist_coeffs,
                args.min_inliers,
            )
            if pseudo is None:
                continue
            accepted.append(
                PseudoLabel(
                    index=int(dataset_index),
                    q_wxyz=q_fused.astype(float).tolist(),
                    t_xyz_m=t_fused.astype(float).tolist(),
                    keypoints_norm=pseudo.tolist(),
                    inliers_a=int(pose_a["inliers"]),
                    inliers_b=int(pose_b["inliers"]),
                    reproj_a_px=float(pose_a["reproj"]),
                    reproj_b_px=float(pose_b["reproj"]),
                    pose_angle_deg=angle,
                    translation_rel=t_rel,
                    keypoint_disagreement_px=kp_disagreement,
                )
            )
        if step == 1 or step % 50 == 0 or step == len(loader):
            logging.info("pseudo step=%d/%d accepted=%d/%d", step, len(loader), len(accepted), len(target_dataset))

    stats = {
        "seen": float(len(target_dataset)),
        "accepted": float(len(accepted)),
        "accepted_pct": float(100.0 * len(accepted) / max(len(target_dataset), 1)),
        "geometry_pass_a_pct": float(100.0 * geometry_pass_a / max(len(target_dataset), 1)),
        "geometry_pass_b_pct": float(100.0 * geometry_pass_b / max(len(target_dataset), 1)),
        "consensus_reject": float(consensus_reject),
        "pose_angle_median_deg": float(np.median([p.pose_angle_deg for p in accepted])) if accepted else float("nan"),
        "translation_rel_median": float(np.median([p.translation_rel for p in accepted])) if accepted else float("nan"),
        "keypoint_disagreement_median_px": (
            float(np.median([p.keypoint_disagreement_px for p in accepted])) if accepted else float("nan")
        ),
    }
    return accepted, stats


def save_pseudo_manifest(round_dir: Path, pseudo_labels: list[PseudoLabel], stats: dict[str, float]) -> None:
    (round_dir / "pseudo_labels.json").write_text(
        json.dumps([asdict(item) for item in pseudo_labels], indent=2),
        encoding="utf-8",
    )
    with (round_dir / "pseudo_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "inliers_a",
                "inliers_b",
                "reproj_a_px",
                "reproj_b_px",
                "pose_angle_deg",
                "translation_rel",
                "keypoint_disagreement_px",
            ],
        )
        writer.writeheader()
        for item in pseudo_labels:
            row = asdict(item)
            writer.writerow({key: row[key] for key in writer.fieldnames})
    (round_dir / "pseudo_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def next_source_batch(source_loader: DataLoader | None, source_iterator):
    if source_loader is None:
        return None, source_iterator
    try:
        batch = next(source_iterator)
    except StopIteration:
        source_iterator = iter(source_loader)
        batch = next(source_iterator)
    return batch, source_iterator


def train_student(
    student: torch.nn.Module,
    target_dataset: Dataset,
    pseudo_labels: list[PseudoLabel],
    source_loader: DataLoader | None,
    args: argparse.Namespace,
    device: torch.device,
    round_index: int,
) -> dict[str, float]:
    trainable = trainable_target_params(student, args.adapt_backbone_norm)
    lr_scale = args.final_lr_scale if round_index == args.rounds else 1.0
    optimizer = torch.optim.AdamW(trainable, lr=args.lr * lr_scale, weight_decay=args.weight_decay)
    pseudo_loader = DataLoader(
        PseudoTargetDataset(target_dataset, pseudo_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    source_iterator = iter(source_loader) if source_loader is not None else None
    student.train()
    losses: list[float] = []
    target_losses: list[float] = []
    source_losses: list[float] = []

    for epoch in range(args.epochs_per_round):
        for step, (images, pseudo_keypoints) in enumerate(pseudo_loader, start=1):
            images = images.to(device, non_blocking=True)
            pseudo_keypoints = pseudo_keypoints.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            target_loss, _ = student(images, pseudo_keypoints)
            total_loss = target_loss

            source_batch, source_iterator = next_source_batch(source_loader, source_iterator)
            if source_batch is not None and args.source_replay_weight > 0:
                source_images, source_keypoints = source_batch
                source_images = source_images.to(device, non_blocking=True)
                source_keypoints = source_keypoints.to(device, non_blocking=True)
                source_loss, _ = student(source_images, source_keypoints)
                total_loss = total_loss + args.source_replay_weight * source_loss
                source_losses.append(float(source_loss.detach().cpu()))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(float(total_loss.detach().cpu()))
            target_losses.append(float(target_loss.detach().cpu()))
            if step == 1 or step % 50 == 0 or step == len(pseudo_loader):
                logging.info(
                    "round=%d epoch=%d step=%d/%d loss=%.4f target=%.4f",
                    round_index,
                    epoch + 1,
                    step,
                    len(pseudo_loader),
                    losses[-1],
                    target_losses[-1],
                )

    return {
        "trainable_parameters": float(sum(parameter.numel() for parameter in trainable)),
        "total_parameters": float(sum(parameter.numel() for parameter in student.parameters())),
        "loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "target_loss_mean": float(np.mean(target_losses)) if target_losses else float("nan"),
        "source_loss_mean": float(np.mean(source_losses)) if source_losses else float("nan"),
        "optimization_steps": float(len(losses)),
        "learning_rate": float(args.lr * lr_scale),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.outdir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S")

    cfg = make_cfg(args, args.dataroot, test_domain=args.domain, test_csv=args.test_csv)
    device = torch.device("cuda:0") if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    set_all_seeds(args.seed, cfg, device.type == "cuda")
    target_dataset = make_target_dataset(cfg, args.max_adapt_samples)
    source_loader = make_source_loader(args)
    camera_matrix, dist_coeffs = load_camera_intrinsics(str(Path(cfg.dataroot) / "camera.json"))
    corners3d = load_tango_3d_keypoints(cfg.keypts_3d_model)

    teacher_a = get_model(cfg)
    teacher_b = get_model(cfg)
    student = get_model(cfg)
    load_weights(teacher_a, args.checkpoint, device)
    load_weights(teacher_b, args.teacher_b_checkpoint or args.checkpoint, device)
    load_weights(student, args.checkpoint, device)

    history: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        round_dir = output_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        pseudo_labels, pseudo_stats = generate_pseudo_labels(
            teacher_a,
            teacher_b,
            target_dataset,
            corners3d,
            camera_matrix,
            dist_coeffs,
            args,
            device,
        )
        save_pseudo_manifest(round_dir, pseudo_labels, pseudo_stats)
        if not pseudo_labels:
            raise RuntimeError(f"Round {round_index}: no pseudo-labels passed the consensus filters")

        student.load_state_dict(teacher_a.state_dict(), strict=True)
        train_stats = train_student(student, target_dataset, pseudo_labels, source_loader, args, device, round_index)
        torch.save(student.state_dict(), round_dir / "model_adapted.pth.tar")
        row = {"round": round_index, **pseudo_stats, **train_stats}
        history.append(row)
        (round_dir / "round_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        logging.info("round summary: %s", row)

        teacher_a.load_state_dict(student.state_dict(), strict=True)
        teacher_b.load_state_dict(student.state_dict(), strict=True)
        teacher_a.eval()
        teacher_b.eval()
        for model in (teacher_a, teacher_b):
            for parameter in model.parameters():
                parameter.requires_grad = False

    torch.save(student.state_dict(), output_root / "model_final.pth.tar")
    (output_root / "adapt_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_root / "adapt_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    logging.info("saved final adapted model to %s", output_root / "model_final.pth.tar")


if __name__ == "__main__":
    main()
