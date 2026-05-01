$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python scripts\build_tlc_fhv_real_dataset.py `
  --output-dir data\tlc_fhv_real_recommended `
  --pickup-start 2022-01-03T08:00:00 `
  --pickup-end 2022-01-03T12:00:00 `
  --row-limit 12000 `
  --num-agents 6 `
  --num-tasks 60 `
  --capacity-scale 1.05 `
  --affinity-penalty-scale 8.0 `
  --resource-penalty-scale 2.0
