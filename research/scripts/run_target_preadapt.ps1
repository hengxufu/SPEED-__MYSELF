param(
  [Parameter(Mandatory = $true)]
  [string]$DataRoot,
  [Parameter(Mandatory = $true)]
  [string]$BaseCheckpoint,
  [ValidateSet("lightbox", "sunlamp")]
  [string]$Domain = "lightbox",
  [string]$RunRoot = "outputs/preadapt"
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

$stage1 = Join-Path $RunRoot "$Domain\stage1"
$stage2 = Join-Path $RunRoot "$Domain\stage2"

& $python research\scripts\target_preadapt_heatmap.py `
  --dataroot $DataRoot --domain $Domain --test_csv test.csv `
  --checkpoint $BaseCheckpoint --outdir $stage1 `
  --epochs 2 --lr 1e-5 --batch_size 16 `
  --adapt_backbone_norm --refresh_teacher_each_epoch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python research\scripts\target_preadapt_heatmap.py `
  --dataroot $DataRoot --domain $Domain --test_csv test.csv `
  --checkpoint (Join-Path $stage1 "model_adapted.pth.tar") --outdir $stage2 `
  --epochs 1 --lr 5e-6 --batch_size 16 `
  --adapt_backbone_norm
exit $LASTEXITCODE
