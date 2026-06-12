from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research" / "results"
FIGURES = ROOT / "research" / "figures"
OUTPUTS = ROOT / "outputs" / "spacecraft_uda"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def parse_results(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(r"^(.+?):\s+([-+0-9.eE]+)\s+\[")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def main() -> None:
    existing = pd.read_csv(RESULTS / "domain_preadapt_summary.csv")
    rows: list[dict] = []
    histories: dict[str, list[dict]] = {}

    for domain in ("lightbox", "sunlamp"):
        baseline = existing[(existing["domain"] == domain) & (existing["method"] == "Synthetic baseline")].iloc[0]
        geometry_final = existing[
            (existing["domain"] == domain) & (existing["method"] == "Iter3+Norm (final)")
        ].iloc[0]
        metrics = parse_results(OUTPUTS / f"eval_{domain}" / "results.txt")
        history = json.loads((OUTPUTS / domain / "adapt_history.json").read_text(encoding="utf-8"))
        histories[domain] = history
        final_round = history[-1]

        methods = [
            (
                "Synthetic baseline",
                float(baseline["speed_thr"]),
                float(baseline["kp_rmse_mean_px"]),
                float(baseline["eT_mean_m"]),
                float(baseline["eR_mean_deg"]),
                float(baseline["ransac_fail_pct"]),
                None,
            ),
            (
                "Spacecraft-UDA-inspired",
                metrics["speed (thr)"],
                metrics["keypoint_rmse_px"],
                metrics["eT"],
                metrics["eR"],
                metrics["pnp_ransac_fail_pct"],
                float(final_round["accepted_pct"]),
            ),
            (
                "Geometry-only final",
                float(geometry_final["speed_thr"]),
                float(geometry_final["kp_rmse_mean_px"]),
                float(geometry_final["eT_mean_m"]),
                float(geometry_final["eR_mean_deg"]),
                float(geometry_final["ransac_fail_pct"]),
                float(geometry_final["pseudo_accepted_pct"]),
            ),
        ]
        for method, speed, rmse, et, er, ransac_fail, accepted in methods:
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "speed_thr": speed,
                    "kp_rmse_mean_px": rmse,
                    "eT_mean_m": et,
                    "eR_mean_deg": er,
                    "ransac_fail_pct": ransac_fail,
                    "pseudo_accepted_pct": accepted,
                    "speed_improve_pct_vs_baseline": 100.0 * (float(baseline["speed_thr"]) - speed)
                    / float(baseline["speed_thr"]),
                    "kp_rmse_improve_pct_vs_baseline": 100.0 * (float(baseline["kp_rmse_mean_px"]) - rmse)
                    / float(baseline["kp_rmse_mean_px"]),
                    "ransac_fail_reduction_pp_vs_baseline": float(baseline["ransac_fail_pct"]) - ransac_fail,
                }
            )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(RESULTS / "spacecraft_uda_preadapt_summary.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "spacecraft_uda_preadapt_summary.json").write_text(
        json.dumps(dataframe.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    for domain, color in (("lightbox", "#386CB0"), ("sunlamp", "#F17C2F")):
        history = histories[domain]
        axes[0].plot(
            [int(row["round"]) for row in history],
            [float(row["accepted_pct"]) for row in history],
            marker="o",
            linewidth=2,
            label=domain,
            color=color,
        )
    axes[0].set_xlabel("UDA round")
    axes[0].set_ylabel("Consensus pseudo-label acceptance (%)")
    axes[0].set_xticks([1, 2, 3])
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    pivot = dataframe.pivot(index="method", columns="domain", values="speed_thr").loc[
        ["Synthetic baseline", "Spacecraft-UDA-inspired", "Geometry-only final"]
    ]
    pivot.plot(kind="bar", ax=axes[1], color=["#386CB0", "#F17C2F"], width=0.72)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Thresholded SPEED score (lower is better)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(title="")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    fig.tight_layout()
    fig.savefig(FIGURES / "spacecraft_uda_acceptance_and_score.png", bbox_inches="tight")
    plt.close(fig)
    print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
