$ErrorActionPreference = "Stop"

Set-Location -Path "D:\进阶项目\CNN\speedplusbaseline-master"

& .\.venv\Scripts\Activate.ps1

python train_overfit_debug.py --dataroot "D:\进阶项目\CNN\speedplusv2" --dataname "speedplus" --domain synthetic --csv "splits_krn/train.csv" `
  --krn_head heatmap --backbone "swin_tiny_patch4_window7_224" --no_backbone_pretrained `
  --num_samples 32 --eval_all_overfit_samples --eval_every 100 --save_pnp_debug --disable_aug `
  --geom_valid_mask --z_min 0.01 `
  --foreground_valid_mask --fg_thr 0.12 --fg_patch 3 `
  --freeze_iters 999999 --weight_decay 0 --iters 1500 --batch_size 8 `
  --savedir "log/overfit_32_strict_geomfg"

