$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python scripts\build_tlc_fhv_real_dataset.py `
  --output-dir data\tlc_fhv_real_large `
  --pickup-start 2022-01-03T08:00:00 `
  --pickup-end 2022-01-03T18:00:00 `
  --row-limit 50000 `
  --num-agents 10 `
  --num-tasks 200 `
  --capacity-scale 1.05 `
  --affinity-penalty-scale 8.0 `
  --resource-penalty-scale 2.0
