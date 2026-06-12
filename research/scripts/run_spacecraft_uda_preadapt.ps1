param(
  [Parameter(Mandatory = $true)]
  [string]$TargetDataRoot,
  [Parameter(Mandatory = $true)]
  [string]$SourceDataRoot,
  [Parameter(Mandatory = $true)]
  [string]$Checkpoint,
  [ValidateSet("lightbox", "sunlamp")]
  [string]$Domain = "lightbox",
  [string]$OutDir = "",
  [int]$Rounds = 3,
  [int]$BatchSize = 16,
  [int]$MaxAdaptSamples = 0,
  [string]$PythonExe = ""
)

$repoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $repoRoot

$pythonCandidates = @()
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
  $pythonCandidates += $PythonExe
}
if (-not [string]::IsNullOrWhiteSpace($env:SPEEDPLUS_PYTHON)) {
  $pythonCandidates += $env:SPEEDPLUS_PYTHON
}
$pythonCandidates += (Join-Path $repoRoot ".venv\Scripts\python.exe")
$pythonCandidates += "python"
$python = $null
foreach ($candidate in $pythonCandidates) {
  try {
    & $candidate -c "import torch, timm, cv2" *> $null
    if ($LASTEXITCODE -eq 0) {
      $python = $candidate
      break
    }
  } catch {}
}
if ($null -eq $python) {
  throw "No usable Python environment with torch, timm, and OpenCV was found."
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = "outputs/spacecraft_uda/$Domain"
}

$argsList = @(
  "research/scripts/spacecraft_uda_preadapt.py",
  "--dataroot", $TargetDataRoot,
  "--source_dataroot", $SourceDataRoot,
  "--domain", $Domain,
  "--checkpoint", $Checkpoint,
  "--outdir", $OutDir,
  "--rounds", "$Rounds",
  "--batch_size", "$BatchSize",
  "--max_adapt_samples", "$MaxAdaptSamples",
  "--adapt_backbone_norm"
)

& $python @argsList
exit $LASTEXITCODE
