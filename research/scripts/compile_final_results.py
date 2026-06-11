from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results"
FIG = ROOT / "research" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


ROWS = [
    ("lightbox", "Synthetic baseline", "outputs/domain_eval/baseline_lightbox_metrics.json", None),
    ("lightbox", "Geo-pseudo v1", "outputs/domain_eval/preadapt_lightbox_metrics.json", "outputs/preadapt/lightbox_geom_pseudo/adapt_history.json"),
    ("lightbox", "Iter2+Norm", "outputs/domain_eval/preadapt_lightbox_iter2_norm_metrics.json", "outputs/preadapt/lightbox_geom_iter2_norm/adapt_history.json"),
    ("lightbox", "Iter3+Norm (final)", "outputs/domain_eval/preadapt_lightbox_iter3_norm_finetune_metrics.json", "outputs/preadapt/lightbox_geom_iter3_norm_finetune/adapt_history.json"),
    ("sunlamp", "Synthetic baseline", "outputs/domain_eval/baseline_sunlamp_metrics.json", None),
    ("sunlamp", "Geo-pseudo v1", "outputs/domain_eval/preadapt_sunlamp_metrics.json", "outputs/preadapt/sunlamp_geom_pseudo/adapt_history.json"),
    ("sunlamp", "Iter2+Norm", "outputs/domain_eval/preadapt_sunlamp_iter2_norm_metrics.json", "outputs/preadapt/sunlamp_geom_iter2_norm/adapt_history.json"),
    ("sunlamp", "Iter2+Norm min5", "outputs/domain_eval/preadapt_sunlamp_iter2_norm_min5_metrics.json", "outputs/preadapt/sunlamp_geom_iter2_norm_min5/adapt_history.json"),
    ("sunlamp", "Iter2+Norm relaxed", "outputs/domain_eval/preadapt_sunlamp_iter2_norm_relaxed_metrics.json", "outputs/preadapt/sunlamp_geom_iter2_norm_relaxed/adapt_history.json"),
    ("sunlamp", "Iter3+Norm (final)", "outputs/domain_eval/preadapt_sunlamp_iter3_norm_finetune_metrics.json", "outputs/preadapt/sunlamp_geom_iter3_norm_finetune/adapt_history.json"),
]


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def last_history(rel: str | None) -> dict:
    if rel is None:
        return {}
    hist = read_json(rel)
    return hist[-1] if hist else {}


def metric_row(domain: str, method: str, metrics_rel: str, hist_rel: str | None) -> dict:
    m = read_json(metrics_rel)
    h = last_history(hist_rel)
    return {
        "domain": domain,
        "method": method,
        "n": int(round(m.get("pnp_eval_cnt", 0))),
        "kp_rmse_mean_px": m.get("keypoint_rmse_px"),
        "kp_rmse_median_px": m.get("keypoint_rmse_px_median"),
        "eT_mean_m": m.get("eT"),
        "eT_median_m": m.get("eT_median"),
        "eR_mean_deg": m.get("eR"),
        "eR_median_deg": m.get("eR_median"),
        "speed_raw": m.get("speed (raw)"),
        "speed_thr": m.get("speed (thr)"),
        "pnp_ok": int(round(m.get("pnp_ok_cnt", 0))),
        "ransac_ok": int(round(m.get("pnp_ransac_ok_cnt", 0))),
        "ransac_fail_pct": m.get("pnp_ransac_fail_pct"),
        "eT_ransac_median_m": m.get("eT_ransac_median"),
        "eR_ransac_median_deg": m.get("eR_ransac_median"),
        "pseudo_accepted": h.get("accepted"),
        "pseudo_accepted_pct": h.get("accepted_pct"),
        "pseudo_reproj_median_px": h.get("pseudo_reproj_median_px"),
        "pseudo_inliers_mean": h.get("pseudo_inliers_mean"),
    }


def add_improvements(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for domain in df["domain"].unique():
        base = df[(df["domain"] == domain) & (df["method"] == "Synthetic baseline")].iloc[0]
        mask = df["domain"] == domain
        df.loc[mask, "speed_raw_improve_pct_vs_baseline"] = (
            (base["speed_raw"] - df.loc[mask, "speed_raw"]) / base["speed_raw"] * 100.0
        )
        df.loc[mask, "kp_rmse_improve_pct_vs_baseline"] = (
            (base["kp_rmse_mean_px"] - df.loc[mask, "kp_rmse_mean_px"]) / base["kp_rmse_mean_px"] * 100.0
        )
        df.loc[mask, "ransac_fail_reduction_pp_vs_baseline"] = (
            base["ransac_fail_pct"] - df.loc[mask, "ransac_fail_pct"]
        )
    return df


def make_horizontal_table(final_df: pd.DataFrame) -> pd.DataFrame:
    lit = [
        ("lightbox", "SPN synthetic-only", "SPEED+ Table 3", 0.45, 65.12, 1.21),
        ("sunlamp", "SPN synthetic-only", "SPEED+ Table 3", 0.65, 92.95, 1.73),
        ("lightbox", "KRN synthetic-only", "SPEED+ Table 3/4", 2.25, 44.53, 1.12),
        ("sunlamp", "KRN synthetic-only", "SPEED+ Table 3/4", 14.64, 80.95, 3.73),
        ("lightbox", "HigherHRNet+EPnP", "SPEED+ Table 3", 0.97, 34.71, 0.77),
        ("sunlamp", "HigherHRNet+EPnP", "SPEED+ Table 3", 0.85, 47.75, 0.98),
        ("lightbox", "KRN+Style Aug.", "SPEED+ Table 4", 1.06, 36.14, 0.81),
        ("sunlamp", "KRN+Style Aug.", "SPEED+ Table 4", 1.32, 62.85, 1.32),
        ("lightbox", "KRN+DANN", "SPEED+ Table 4", 0.95, 33.62, 0.74),
        ("sunlamp", "KRN+DANN", "SPEED+ Table 4", 2.04, 65.37, 1.47),
        ("lightbox", "Oracle", "SPEED+ Table 4", 0.24, 6.15, 0.15),
        ("sunlamp", "Oracle", "SPEED+ Table 4", 0.19, 5.33, 0.13),
    ]
    rows = [
        {
            "domain": d,
            "method": method,
            "source": src,
            "eT_m": et,
            "eR_deg": er,
            "epose": ep,
            "note": "Published SPEED+ benchmark; HIL pose score is reported as in the source paper.",
        }
        for d, method, src, et, er, ep in lit
    ]
    ours = final_df[final_df["method"].isin(["Synthetic baseline", "Iter3+Norm (final)"])]
    for _, r in ours.iterrows():
        rows.append(
            {
                "domain": r["domain"],
                "method": "Ours baseline" if r["method"] == "Synthetic baseline" else "Ours final",
                "source": "This work",
                "eT_m": r["eT_mean_m"],
                "eR_deg": r["eR_mean_deg"],
                "epose": r["speed_thr"],
                "note": "Swin heatmap keypoint model; thresholded SPEED score from local evaluation script.",
            }
        )
    return pd.DataFrame(rows)


def plot_domain_curves(df: pd.DataFrame) -> None:
    selected = df[df["method"].isin(["Synthetic baseline", "Geo-pseudo v1", "Iter2+Norm", "Iter3+Norm (final)"])].copy()
    order = ["Synthetic baseline", "Geo-pseudo v1", "Iter2+Norm", "Iter3+Norm (final)"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=180)
    for ax, metric, ylabel in [
        (axes[0], "speed_raw", "SPEED score (raw, lower is better)"),
        (axes[1], "ransac_fail_pct", "RANSAC failure rate (%)"),
    ]:
        pivot = selected.pivot(index="method", columns="domain", values=metric).loc[order]
        pivot.plot(kind="bar", ax=ax, width=0.74, color=["#386cb0", "#f17c2f"])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="")
        ax.tick_params(axis="x", rotation=18)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(FIG / "domain_preadapt_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_horizontal(horizontal: pd.DataFrame) -> None:
    methods = [
        "KRN synthetic-only",
        "KRN+Style Aug.",
        "KRN+DANN",
        "HigherHRNet+EPnP",
        "Ours baseline",
        "Ours final",
        "Oracle",
    ]
    plot_df = horizontal[horizontal["method"].isin(methods)].copy()
    pivot = plot_df.pivot(index="method", columns="domain", values="epose").loc[methods]
    fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=180)
    pivot.plot(kind="bar", ax=ax, width=0.76, color=["#386cb0", "#f17c2f"])
    ax.set_ylabel("Pose error / SPEED score (lower is better)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    ax.tick_params(axis="x", rotation=22)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(FIG / "speedplus_horizontal_epose.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.DataFrame(metric_row(*row) for row in ROWS)
    df = add_improvements(df)
    df.to_csv(OUT / "domain_preadapt_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "domain_preadapt_summary.json").write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    horizontal = make_horizontal_table(df)
    horizontal.to_csv(OUT / "speedplus_horizontal_comparison.csv", index=False, encoding="utf-8-sig")
    (OUT / "speedplus_horizontal_comparison.json").write_text(
        json.dumps(horizontal.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_domain_curves(df)
    plot_horizontal(horizontal)
    print(df[["domain", "method", "kp_rmse_mean_px", "eT_mean_m", "eR_mean_deg", "speed_raw", "speed_raw_improve_pct_vs_baseline", "ransac_fail_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
