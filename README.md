# Teoria da Decisão - Tomada de decisão multicritério

Disciplina **Teoria da Decisão** contendo os códigos e artefatos das três entregas (otimização mono-objetivo, multiobjetivo e decisão multicritério).

## Estrutura
- `data/` — instância do problema: `custos.csv`, `recursos.csv`, `capacidades.csv`.
- `src/` — scripts Python para cada etapa (`otimizacao_mono_objetivo.py`, `otimizacao_multiobjetivo.py`, `decisao_multicriterio.py`).
- `graphs/` — gráficos e tabelas gerados pelos scripts, separados por etapa.
- `cod.tex` — fonte LaTeX do relatório (usa as figuras em `graphs/`).
- PDFs prontos: `Relatório TC3...pdf` e `Apresentação TC3...pdf`.

## Requisitos
- Python 3.10+ e `pip`.
- Instale as dependências:  
  ```bash
  pip install -r requirements.txt
  ```

## Como reproduzir
Os scripts usam sementes fixas para garantir repetibilidade básica.

1) **Otimização mono-objetivo** (`f1` e `f2`):  
   ```bash
   python src/otimizacao_mono_objetivo.py
   ```  
   Saídas em `graphs/otimizacao_mono_objetivo/` (`f1_convergencia.png`, `f1_melhor.png`, etc.).

2) **Otimização multiobjetivo** (Soma Ponderada e ε-Restrito):  
   ```bash
   python src/otimizacao_multiobjetivo.py
   ```  
   Gera as fronteiras em `graphs/otimizacao_multiobjetivo/`.

3) **Decisão multicritério** (agregação clássica + TOPSIS) a partir das fronteiras multiobjetivo:  
   ```bash
   python src/decisao_multicriterio.py
   ```  
   Resultados consolidados em `graphs/decisao_multicriterio/decisao_resumo.csv` e gráficos correspondentes.  
   Para rodar mais rápido, defina `ENTREGA3_FAST=1` antes do comando.