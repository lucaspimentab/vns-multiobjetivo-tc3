Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python scripts\build_tlc_fhv_balanced_subset.py `
  --source-dir data\tlc_fhv_real_large `
  --output-dir data\tlc_fhv_real_balanced `
  --max-count-ratio 2.0 `
  --min-agents 9 `
  --seed 42

python src\sbpo_pipeline.py `
  --data-dir data\tlc_fhv_real_balanced `
  --dataset-label tlc_fhv_real_balanced
