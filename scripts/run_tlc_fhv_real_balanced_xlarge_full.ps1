Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python scripts\build_tlc_fhv_real_dataset.py `
  --output-dir data\tlc_fhv_real_balanced_xlarge `
  --pickup-start 2022-01-03T00:00:00 `
  --pickup-end 2022-01-03T23:59:59 `
  --row-limit 120000 `
  --num-agents 15 `
  --num-tasks 300 `
  --skip-top-bases 1 `
  --capacity-scale 1.05 `
  --affinity-penalty-scale 8.0 `
  --resource-penalty-scale 2.0 `
  --seed 42

python src\sbpo_pipeline.py `
  --data-dir data\tlc_fhv_real_balanced_xlarge `
  --dataset-label tlc_fhv_real_balanced_xlarge
