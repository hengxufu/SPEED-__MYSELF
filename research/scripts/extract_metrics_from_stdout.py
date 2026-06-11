from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stdout")
    parser.add_argument("out_json")
    args = parser.parse_args()

    text = read_text(Path(args.stdout))
    start = text.rfind("\n{")
    if start < 0:
        start = text.find("{")
    if start < 0:
        raise SystemExit(f"No JSON object found in {args.stdout}")
    snippet = text[start + 1 :].strip()
    end = snippet.rfind("}")
    if end < 0:
        raise SystemExit(f"No closing JSON brace found in {args.stdout}")
    metrics = json.loads(snippet[: end + 1])
    Path(args.out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_json": args.out_json,
        "eT": metrics.get("eT"),
        "eR": metrics.get("eR"),
        "speed_raw": metrics.get("speed (raw)"),
        "kp_rmse": metrics.get("keypoint_rmse_px"),
        "ransac_fail": metrics.get("pnp_ransac_fail_pct"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
