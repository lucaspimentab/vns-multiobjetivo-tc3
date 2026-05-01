Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python scripts\build_tlc_fhv_balanced_subset.py `
  --source-dir data\tlc_fhv_real_balanced_xlarge `
  --output-dir data\tlc_fhv_real_diag_10x120 `
  --max-count-ratio 1.8 `
  --exact-agents 10 `
  --tasks-per-agent 12 `
  --seed 42
