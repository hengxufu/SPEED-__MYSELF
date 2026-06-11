from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))

from src.utils.utils import load_camera_intrinsics, load_tango_3d_keypoints, project_keypoints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create bbox-conditioned SPEED+ target CSV files for evaluation and pre-adaptation."
    )
    parser.add_argument("--dataroot", required=True, help="SPEED+ root containing camera.json and domain folders")
    parser.add_argument(
        "--outroot",
        default=str(REPO_ROOT / "work" / "target_dataset"),
        help="Output root for generated target CSV files",
    )
    return parser.parse_args()


def _absolute_image(data_root: Path, domain: str, filename: str) -> str:
    return str((data_root / domain / "images" / filename).resolve())


def prepare_lightbox(data_root: Path, out_root: Path) -> None:
    src = data_root / "lightbox" / "splits_krn" / "test.csv"
    dst = out_root / "lightbox" / "splits_krn" / "test.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8", newline="") as f_in, dst.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        for row in reader:
            if not row:
                continue
            rel = row[0].strip().replace("\\", "/")
            filename = Path(rel).name
            row[0] = _absolute_image(data_root, "lightbox", filename)
            writer.writerow(row)


def prepare_sunlamp(data_root: Path, out_root: Path) -> None:
    dst = out_root / "sunlamp" / "splits_krn" / "test.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)

    labels = json.loads((data_root / "sunlamp" / "test.json").read_text(encoding="utf-8"))
    camera_matrix, dist_coeffs = load_camera_intrinsics(str(data_root / "camera.json"))
    keypts3d = load_tango_3d_keypoints(str(REPO_ROOT / "src" / "utils" / "tangoPoints.mat"))

    with dst.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        for item in labels:
            filename = item["filename"]
            q = np.array(item["q_vbs2tango_true"], dtype=np.float32)
            t = np.array(item["r_Vo2To_vbs_true"], dtype=np.float32)
            keypts2d = project_keypoints(q, t, camera_matrix, dist_coeffs, keypts3d)
            bbox = [
                float(np.amin(keypts2d[0])),
                float(np.amax(keypts2d[0])),
                float(np.amin(keypts2d[1])),
                float(np.amax(keypts2d[1])),
            ]
            keypts_flat = np.reshape(np.transpose(keypts2d), (22,)).astype(float).tolist()
            row = [_absolute_image(data_root, "sunlamp", filename)] + bbox + q.tolist() + t.tolist() + keypts_flat
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    data_root = Path(args.dataroot).expanduser().resolve()
    out_root = Path(args.outroot).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_root / "camera.json", out_root / "camera.json")
    prepare_lightbox(data_root, out_root)
    prepare_sunlamp(data_root, out_root)

    for domain in ("lightbox", "sunlamp"):
        csv_path = out_root / domain / "splits_krn" / "test.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            count = sum(1 for _ in f)
        print(f"{domain}: {count} rows -> {csv_path}")


if __name__ == "__main__":
    main()
