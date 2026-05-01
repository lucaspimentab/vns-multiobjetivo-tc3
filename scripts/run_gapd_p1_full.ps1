$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python src\sbpo_pipeline.py `
  --orlib-file data\gapd.txt `
  --problem-index 1 `
  --dataset-label gapd_p1
