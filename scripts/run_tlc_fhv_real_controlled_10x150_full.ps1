Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python scripts\build_tlc_fhv_balanced_subset.py `
  --source-dir data\tlc_fhv_real_balanced_xlarge `
  --output-dir data\tlc_fhv_real_controlled_10x150 `
  --max-count-ratio 1.8 `
  --exact-agents 10 `
  --target-total-tasks 150 `
  --min-tasks-per-agent 12 `
  --max-tasks-per-agent 18 `
  --seed 42

python src\sbpo_pipeline.py `
  --data-dir data\tlc_fhv_real_controlled_10x150 `
  --dataset-label tlc_fhv_real_controlled_10x150
