param(
  [int]$MaxEpochs = 50,
  [int]$BatchSize = 48
)

Set-Location $PSScriptRoot\..

& .\.venv\Scripts\python -u train_probe_heatmap.py `
  --model_name krn --krn_head heatmap `
  --backbone swin_tiny_patch4_window7_224 `
  --backbone_pretrained_path log/pretrained/swin_tiny_patch4_window7_224.pth `
  --heatmap_loss heatmap_ce_coord_aux --coord_aux_weight 0.2 `
  --heatmap_decode softargmax `
  --heatmap_hard_kpts 3,4,5,6,10 --heatmap_hard_kpt_weight 1.5 --heatmap_other_kpt_weight 1.0 `
  --backbone_fpn --backbone_out_indices 1,2,3 `
  --freeze_backbone_epochs 5 `
  --optimizer adamw --lr_schedule cosine --weight_decay 0.05 --warmup_epochs 5 --min_lr 0 `
  --lr_backbone 1e-5 --lr_head 1e-4 `
  --input_shape 224 224 --heatmap_size 56 56 `
  --deterministic_crop --p_aug 0 `
  --train_subset_size 512 --val_subset_size 500 `
  --ckpt_every 5 `
  --train_domain synthetic --test_domain synthetic --train_csv train.csv --test_csv validation.csv `
  --max_epochs $MaxEpochs --batch_size $BatchSize --num_workers 4 `
  --logdir log/baseline_224_56_long

