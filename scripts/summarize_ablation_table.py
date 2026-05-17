import argparse
import os
import pandas as pd


def _last_row(path):
    df = pd.read_csv(path, sep='\t')
    if 'epoch' in df.columns:
        df = df.sort_values('epoch')
    return df.iloc[-1], df.columns


def _mean_of_prefix(row, cols, prefix):
    xs = [c for c in cols if c.startswith(prefix)]
    if not xs:
        return float('nan')
    vals = []
    for c in xs:
        try:
            vals.append(float(row[c]))
        except Exception:
            pass
    return float(sum(vals) / len(vals)) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdirs', nargs='+', required=True)
    args = ap.parse_args()

    rows = []
    for ld in args.logdirs:
        p = os.path.join(ld, 'probe_results.txt')
        if not os.path.exists(p):
            rows.append({'exp': os.path.basename(ld), 'missing': True})
            continue
        last, cols = _last_row(p)
        rows.append(
            {
                'exp': os.path.basename(ld),
                'epoch': int(last.get('epoch', -1)),
                'val_rmse_med_px': float(last.get('val_keypoint_rmse_px_median', float('nan'))),
                'val_rmse_mean_px': float(last.get('val_keypoint_rmse_px', float('nan'))),
                'val_rmse_norm_input_med': float(last.get('val_keypoint_rmse_norm_input_median', float('nan'))),
                'val_rmse_norm_bbox_med': float(last.get('val_keypoint_rmse_norm_bbox_median', float('nan'))),
                'val_pck5_mean': _mean_of_prefix(last, cols, 'val_per_keypoint_pck5_pct_'),
                'val_pck10_mean': _mean_of_prefix(last, cols, 'val_per_keypoint_pck10_pct_'),
                'val_pck05bbox_mean': _mean_of_prefix(last, cols, 'val_per_keypoint_pck05_bbox_pct_'),
                'val_pck10bbox_mean': _mean_of_prefix(last, cols, 'val_per_keypoint_pck10_bbox_pct_'),
                'val_peak_ratio_mean': _mean_of_prefix(last, cols, 'val_per_keypoint_peak_ratio_'),
                'val_ransac_ok_cnt': float(last.get('val_pnp_ransac_ok_cnt', float('nan'))),
                'val_ransac_inlier_med': float(last.get('val_pnp_ransac_inlier_cnt_median', float('nan'))),
                'val_reproj_med_px': float(last.get('val_reprojection_error_median_px', float('nan'))),
                'val_pose_valid_cnt': float(last.get('val_pose_valid_cnt', float('nan'))),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(['missing', 'exp'])
    with pd.option_context('display.max_columns', 200, 'display.width', 200):
        print(out.to_string(index=False))


if __name__ == '__main__':
    main()

