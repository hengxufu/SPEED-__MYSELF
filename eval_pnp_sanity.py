import os.path as osp
import argparse
import numpy as np
import pandas as pd

from src.utils.utils import load_tango_3d_keypoints, load_camera_intrinsics, pnp
from src.utils.metrics import error_orientation, error_translation, speed_score


def main():
    ap = argparse.ArgumentParser('PnP sanity check with GT 2D keypoints')
    ap.add_argument('--dataroot', type=str, required=True)
    ap.add_argument('--dataname', type=str, default='')
    ap.add_argument('--domain', type=str, default='synthetic')
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--keypts_3d_model', type=str, default='src/utils/tangoPoints.mat')
    ap.add_argument('--camera_json', type=str, default='camera.json')
    ap.add_argument('--num', type=int, default=50)
    args = ap.parse_args()

    root = osp.join(args.dataroot, args.dataname) if args.dataname else args.dataroot
    csv_path = osp.join(root, args.domain, args.csv) if not osp.isabs(args.csv) else args.csv
    cam_path = osp.join(root, args.camera_json) if not osp.isabs(args.camera_json) else args.camera_json
    kp3d_path = osp.join(osp.dirname(__file__), args.keypts_3d_model) if not osp.isabs(args.keypts_3d_model) else args.keypts_3d_model

    df = pd.read_csv(csv_path, header=None)
    corners3D = load_tango_3d_keypoints(kp3d_path)
    cameraMatrix, distCoeffs = load_camera_intrinsics(cam_path)

    keypts_2d_all = []
    keypts_3d_all = corners3D.reshape(-1, 3)

    print('cameraMatrix=\n', cameraMatrix)
    print('distCoeffs=\n', distCoeffs.reshape(-1))
    print('3D keypoints min/max=', keypts_3d_all.min(axis=0), keypts_3d_all.max(axis=0))
    print('csv=', csv_path)
    print('num_rows=', len(df))

    n = min(int(args.num), len(df))
    eT_list, eR_list, sp_list = [], [], []
    for i in range(n):
        row = df.iloc[i].to_numpy()
        q_gt = row[5:9].astype(np.float32)
        t_gt = row[9:12].astype(np.float32)

        k = row[12:].astype(np.float32)
        k = k.reshape(-1, 2)  # (11,2) pixel coords
        keypts_2d_all.append(k)

        q_pr, t_pr = pnp(corners3D, k, cameraMatrix, distCoeffs)
        eR = error_orientation(q_pr, q_gt)
        eT = error_translation(t_pr, t_gt)
        sp_raw, _ = speed_score(t_pr, q_pr, t_gt, q_gt, applyThresh=False)

        eR_list.append(float(eR))
        eT_list.append(float(eT))
        sp_list.append(float(sp_raw))

        if i < 3:
            print(f'--- sample {i} ---')
            print('2D keypoints min/max=', k.min(axis=0), k.max(axis=0))
            print('t_gt=', t_gt, 't_pr=', t_pr)
            print('q_gt=', q_gt, 'q_pr=', q_pr)
            print('eT=', eT, 'eR=', eR, 'speed_raw=', sp_raw)

    keypts_2d_all = np.concatenate(keypts_2d_all, axis=0)
    print('2D keypoints global min/max=', keypts_2d_all.min(axis=0), keypts_2d_all.max(axis=0))
    print('eT mean/median=', float(np.mean(eT_list)), float(np.median(eT_list)))
    print('eR mean/median=', float(np.mean(eR_list)), float(np.median(eR_list)))
    print('speed_raw mean/median=', float(np.mean(sp_list)), float(np.median(sp_list)))


if __name__ == '__main__':
    main()

