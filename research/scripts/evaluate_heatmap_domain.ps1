param(
  [Parameter(Mandatory = $true)]
  [string]$DataRoot,
  [Parameter(Mandatory = $true)]
  [string]$Checkpoint,
  [ValidateSet("lightbox", "sunlamp", "synthetic")]
  [string]$Domain = "lightbox",
  [string]$LogDir = "outputs/domain_eval/latest",
  [int]$BatchSize = 32
)

$repoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = "python"
if (Test-Path $venvPython) {
  & $venvPython -c "import sys" *> $null
  if ($LASTEXITCODE -eq 0) {
    $python = $venvPython
  }
}

$argsList = @(
  "test.py",
  "--projroot", $repoRoot,
  "--dataroot", $DataRoot,
  "--model_name", "krn",
  "--krn_head", "heatmap",
  "--backbone", "swin_tiny_patch4_window7_224",
  "--backbone_fpn",
  "--backbone_out_indices", "1,2,3",
  "--heatmap_loss", "heatmap_ce_coord_aux",
  "--coord_aux_weight", "0.2",
  "--heatmap_decode", "softargmax",
  "--input_shape", "224", "224",
  "--heatmap_size", "56", "56",
  "--deterministic_crop",
  "--test_domain", $Domain,
  "--test_csv", "test.csv",
  "--pretrained", $Checkpoint,
  "--logdir", $LogDir,
  "--batch_size", "$BatchSize",
  "--num_workers", "4"
)

& $python @argsList
exit $LASTEXITCODE
