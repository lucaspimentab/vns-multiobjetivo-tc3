# SBPO 2026 - GRASP/VNS + FITradeoff

Este repositorio foi reorganizado para servir como base de um artigo/aplicacao ao SBPO sobre uma ferramenta de apoio a decisao para o problema multiobjetivo de atribuicao de tarefas a agentes com restricoes de capacidade.

A narrativa principal agora e:

- geracao de alternativas nao-dominadas com `GRASP + VNS`;
- uso de apenas uma estrategia escalar de otimizacao multiobjetivo: `epsilon-restrito`;
- selecao final com `FITradeoff`, em vez de agregacao ponderada classica e TOPSIS.

## Pipeline principal

O script principal do projeto agora e:

```bash
python src/sbpo_pipeline.py
```

Para uma validacao mais curta:

```bash
python src/sbpo_pipeline.py --fast
```

Saidas geradas para a instancia local completa:

- `graphs/sbpo_fitradeoff/instancia_local/epsilon_frontier_all.csv`
- `graphs/sbpo_fitradeoff/instancia_local/fitradeoff_candidates.csv`
- `graphs/sbpo_fitradeoff/instancia_local/fitradeoff_questions.csv`
- `graphs/sbpo_fitradeoff/instancia_local/fitradeoff_weights.csv`
- `graphs/sbpo_fitradeoff/instancia_local/fitradeoff_witness_weights.csv`
- `graphs/sbpo_fitradeoff/instancia_local/fitradeoff_fronteira.png`
- `graphs/sbpo_fitradeoff/instancia_local/fitradeoff_cargas.png`
- `graphs/sbpo_fitradeoff/instancia_local/fitradeoff_intervalos_pesos.png`

## Dataset novo

Para rodar o benchmark maior `gapd`, problema 1, em modo completo:

```bash
python src/sbpo_pipeline.py --orlib-file data/gapd.txt --problem-index 1 --dataset-label gapd_p1
```

Ou via script PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gapd_p1_full.ps1
```

As saidas desse benchmark vao para:

- `graphs/sbpo_fitradeoff/gapd_p1/`

## Dataset real oficial

Tambem foi preparado um fluxo com dados reais oficiais da TLC (New York City Taxi and Limousine Commission), adaptados para o problema de atribuicao com capacidade.

Fontes usadas:

- viagens FHV 2022: `https://data.cityofnewyork.us/d/vgi6-tcdb`
- bases TLC vigentes: `https://data.cityofnewyork.us/api/views/eccv-9dzr/rows.csv?accessType=DOWNLOAD`
- tabela de zonas TLC: `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`

Para construir a instancia real recomendada:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_tlc_fhv_real_recommended.ps1
```

Para construir e rodar tudo em modo completo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tlc_fhv_real_full.ps1
```

Ou manualmente:

```bash
python scripts/build_tlc_fhv_real_dataset.py --output-dir data/tlc_fhv_real_recommended --pickup-start 2022-01-03T08:00:00 --pickup-end 2022-01-03T12:00:00 --row-limit 12000 --num-agents 6 --num-tasks 60 --capacity-scale 1.05 --affinity-penalty-scale 8.0 --resource-penalty-scale 2.0
python src/sbpo_pipeline.py --data-dir data/tlc_fhv_real_recommended --dataset-label tlc_fhv_real_recommended
```

Arquivos gerados da instancia real:

- `data/tlc_fhv_real_recommended/custos.csv`
- `data/tlc_fhv_real_recommended/recursos.csv`
- `data/tlc_fhv_real_recommended/capacidades.csv`
- `data/tlc_fhv_real_recommended/agents.csv`
- `data/tlc_fhv_real_recommended/tasks.csv`
- `data/tlc_fhv_real_recommended/metadata.json`

## Dataset real grande

Tambem foi preparada uma variante real bem maior:

- `10` agentes reais (bases TLC)
- `200` tarefas reais (viagens FHV)
- janela de coleta: `2022-01-03 08:00` ate `2022-01-03 18:00`

Para construir essa instancia:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_tlc_fhv_real_large.ps1
```

Para rodar tudo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tlc_fhv_real_large_full.ps1
```

Saidas esperadas:

- `data/tlc_fhv_real_large/`
- `graphs/sbpo_fitradeoff/tlc_fhv_real_large/`

## Dataset real balanceado

Tambem foi preparado um subconjunto menos desbalanceado, ainda derivado de dados reais oficiais, a partir da instancia `tlc_fhv_real_large`.

Ideia:

- remove a base dominante que distorce a capacidade;
- seleciona o maior subconjunto de bases com contagens amostrais semelhantes;
- amostra o mesmo numero de tarefas por base;
- recalcula as capacidades com a mesma logica da instancia real original.

Para construir essa instancia balanceada:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_tlc_fhv_real_balanced.ps1
```

Para construir e rodar tudo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tlc_fhv_real_balanced_full.ps1
```

Ou manualmente:

```bash
python scripts/build_tlc_fhv_balanced_subset.py --source-dir data/tlc_fhv_real_large --output-dir data/tlc_fhv_real_balanced --max-count-ratio 2.0 --min-agents 9
python src/sbpo_pipeline.py --data-dir data/tlc_fhv_real_balanced --dataset-label tlc_fhv_real_balanced
```

Saidas esperadas:

- `data/tlc_fhv_real_balanced/`
- `graphs/sbpo_fitradeoff/tlc_fhv_real_balanced/`

Se quiser uma versao maior, removendo apenas a base dominante e mantendo todas as tarefas das demais bases selecionadas:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tlc_fhv_real_balanced_large_full.ps1
```

Ou manualmente:

```bash
python scripts/build_tlc_fhv_balanced_subset.py --source-dir data/tlc_fhv_real_large --output-dir data/tlc_fhv_real_balanced_large --max-count-ratio 2.0 --min-agents 9 --keep-all-tasks
python src/sbpo_pipeline.py --data-dir data/tlc_fhv_real_balanced_large --dataset-label tlc_fhv_real_balanced_large
```

Se quiser crescer de verdade, reconstruindo direto da base bruta da TLC e ignorando a base mais dominante na selecao dos agentes:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tlc_fhv_real_balanced_xlarge_full.ps1
```

Ou manualmente:

```bash
python scripts/build_tlc_fhv_real_dataset.py --output-dir data/tlc_fhv_real_balanced_xlarge --pickup-start 2022-01-03T00:00:00 --pickup-end 2022-01-03T23:59:59 --row-limit 120000 --num-agents 15 --num-tasks 300 --skip-top-bases 1 --capacity-scale 1.05 --affinity-penalty-scale 8.0 --resource-penalty-scale 2.0
python src/sbpo_pipeline.py --data-dir data/tlc_fhv_real_balanced_xlarge --dataset-label tlc_fhv_real_balanced_xlarge
```

## Dataset controlado 10x150

Para um experimento real mais administravel e menos trivial, foi preparado um caso intermediario com:

- `10` agentes;
- `150` tarefas;
- cotas controladas por base entre `12` e `18` tarefas;
- selecao de bases com razao max/min original limitada por `1.8`.

Para construir:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_tlc_fhv_real_controlled_10x150.ps1
```

Para rodar completo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tlc_fhv_real_controlled_10x150_full.ps1
```

Ou manualmente:

```bash
python scripts/build_tlc_fhv_balanced_subset.py --source-dir data/tlc_fhv_real_balanced_xlarge --output-dir data/tlc_fhv_real_controlled_10x150 --max-count-ratio 1.8 --exact-agents 10 --target-total-tasks 150 --min-tasks-per-agent 12 --max-tasks-per-agent 18
python src/sbpo_pipeline.py --data-dir data/tlc_fhv_real_controlled_10x150 --dataset-label tlc_fhv_real_controlled_10x150
```

## Dataset diagnostico 10x120

Se quiser manter o caso mais leve e administravel, use o dataset `10 x 120` com `12` tarefas por agente:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tlc_fhv_real_diag_10x120_full.ps1
```

Ou manualmente:

```bash
python scripts/build_tlc_fhv_balanced_subset.py --source-dir data/tlc_fhv_real_balanced_xlarge --output-dir data/tlc_fhv_real_diag_10x120 --max-count-ratio 1.8 --exact-agents 10 --tasks-per-agent 12
python src/sbpo_pipeline.py --data-dir data/tlc_fhv_real_diag_10x120 --dataset-label tlc_fhv_real_diag_10x120
```

## Estrutura

- `data/`: instancia do problema.
- `src/sbpo_pipeline.py`: pipeline SBPO com VNS/GRASP, epsilon-restrito e FITradeoff.
- `graphs/sbpo_fitradeoff/`: resultados por dataset.
- `scripts/run_gapd_p1_full.ps1`: execucao completa pronta para o benchmark maior.
- `docs/`: PDFs antigos da disciplina e um guia editavel para a adaptacao SBPO.

## Perfil FITradeoff

O script usa quatro criterios na etapa de decisao:

1. `f1`: custo total.
2. `f2`: desequilibrio de utilizacao relativa entre agentes.
3. `slack_min`: folga minima residual.
4. `cost_var`: variacao media de custo sob perturbacoes.

As perguntas FITradeoff sao geradas a partir de uma ordem de importancia coerente com o perfil originalmente usado no trabalho:

`f1 > f2 > slack_min >= cost_var`

No estado atual, o repositorio usa um perfil sintetico e reproduzivel para construir a sessao FITradeoff. Isso e suficiente para o artigo e para os experimentos computacionais. Se depois houver um decisor real, basta trocar esse perfil pelas respostas reais na configuracao do script.

## Scripts legados

Os scripts abaixo foram mantidos como referencia do trabalho da disciplina:

- `src/otimizacao_mono_objetivo.py`
- `src/otimizacao_multiobjetivo.py`
- `src/decisao_multicriterio.py`

Eles nao sao mais o fluxo recomendado para o material do SBPO.

## Dependencias

Instalacao minima:

```bash
pip install -r requirements.txt
```

O novo pipeline tambem usa `scipy`. Se necessario:

```bash
pip install scipy
```
