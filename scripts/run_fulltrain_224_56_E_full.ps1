param(
  [int]$MaxEpochs = 75,
  [int]$BatchSize = 48,
  [int]$TestEvery = 1,
  [string]$RunTag = "run1"
)

Set-Location $PSScriptRoot\..

$logdir = "log/fulltrain_224_56_E_full_$RunTag"
$savedir = "checkpoints/fulltrain_224_56_E_full_$RunTag"

$argsList = @(
  '-u', 'train.py',
  '--model_name', 'krn', '--krn_head', 'heatmap',
  '--backbone', 'swin_tiny_patch4_window7_224',
  '--backbone_pretrained_path', 'checkpoints/pretrained/swin_tiny_patch4_window7_224.pth',
  '--backbone_fpn', '--backbone_out_indices', '1,2,3',
  '--heatmap_loss', 'heatmap_ce_coord_aux', '--coord_aux_weight', '0.2',
  '--heatmap_decode', 'softargmax',
  '--heatmap_hard_kpts', '3,4,5,6,10', '--heatmap_hard_kpt_weight', '1.5', '--heatmap_other_kpt_weight', '1.0',
  '--freeze_backbone_epochs', '5',
  '--optimizer', 'adamw', '--lr_schedule', 'cosine', '--weight_decay', '0.05', '--warmup_epochs', '5', '--min_lr', '0',
  '--lr_backbone', '1e-5', '--lr_head', '1e-4',
  '--input_shape', '224', '224', '--heatmap_size', '56', '56',
  '--deterministic_crop', '--p_aug', '0',
  '--train_domain', 'synthetic', '--test_domain', 'synthetic', '--train_csv', 'train.csv', '--test_csv', 'validation.csv',
  '--max_epochs', "$MaxEpochs", '--batch_size', "$BatchSize", '--num_workers', '4',
  '--test_epoch', "$TestEvery",
  '--debug_save_n', '50',
  '--start_over',
  '--savedir', $savedir,
  '--logdir', $logdir
)

& .\.venv\Scripts\python @argsList
