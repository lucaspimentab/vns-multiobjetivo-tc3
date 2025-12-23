import os
import random
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1) Dados
try:
    c = pd.read_csv("data/custos.csv", header=None).values
    a = pd.read_csv("data/recursos.csv", header=None).values
    b = pd.read_csv("data/capacidades.csv", header=None).values.flatten()
except FileNotFoundError:
    print("Arquivos de dados nao encontrados. Gerando dados aleatorios para teste.")
    m, n = 5, 50
    c = np.random.rand(m, n) * 10
    a = np.random.rand(m, n) * 5
    b = np.full(m, 100)
else:
    m, n = c.shape


# 2) Funções objetivo e viabilidade
def f1(sol: np.ndarray) -> float:
    """Custo total (minimizacao)."""
    return float(sum(c[sol[j], j] for j in range(n)))


def f2(sol: np.ndarray) -> float:
    """Desequilibrio de carga (minimizacao). Retorna inf se inviavel."""
    load = np.zeros(m)
    for j in range(n):
        load[sol[j]] += a[sol[j], j]
    if np.any(load > b):
        return float("inf")
    return float(np.max(load) - np.min(load))


def is_feasible(sol: np.ndarray) -> bool:
    load = np.zeros(m)
    for j in range(n):
        load[sol[j]] += a[sol[j], j]
    return bool(np.all(load <= b))


def compute_loads(sol: np.ndarray) -> np.ndarray:
    loads = np.zeros(m)
    for j in range(n):
        loads[sol[j]] += a[sol[j], j]
    return loads


# 3) Heuristica construtiva (GRASP)
def greedy_solution_grasp(alpha: float = 0.3) -> np.ndarray:
    sol = -np.ones(n, dtype=int)
    load = np.zeros(m)
    tasks = list(range(n))
    random.shuffle(tasks)

    for j in tasks:
        feasible_agents = []
        for i in range(m):
            if load[i] + a[i, j] <= b[i]:
                feasible_agents.append((c[i, j], i))

        if not feasible_agents:
            return None

        feasible_agents.sort()
        min_cost = feasible_agents[0][0]
        max_cost_limit = min_cost + alpha * (feasible_agents[-1][0] - min_cost)
        rcl = [i for cost, i in feasible_agents if cost <= max_cost_limit]
        if not rcl:
            rcl = [feasible_agents[0][1]]

        chosen = random.choice(rcl)
        sol[j] = chosen
        load[chosen] += a[chosen, j]

    return sol


# 4) Estruturas de vizinhanca
def neighborhood_shift(sol: np.ndarray) -> np.ndarray:
    s = sol.copy()
    j = random.randrange(n)
    current = s[j]
    new_agent = random.randint(0, m - 1)
    while new_agent == current:
        new_agent = random.randint(0, m - 1)
    s[j] = new_agent
    return s


def neighborhood_exchange(sol: np.ndarray) -> np.ndarray:
    s = sol.copy()
    j1, j2 = random.sample(range(n), 2)
    s[j1], s[j2] = s[j2], s[j1]
    return s


def neighborhood_swap(sol: np.ndarray) -> np.ndarray:
    s = sol.copy()
    agents = list(set(s))
    if len(agents) < 2:
        return s

    i1, i2 = random.sample(agents, 2)
    tasks_i1 = np.where(s == i1)[0]
    tasks_i2 = np.where(s == i2)[0]
    if len(tasks_i1) == 0 or len(tasks_i2) == 0:
        return s

    j1 = random.choice(tasks_i1)
    j2 = random.choice(tasks_i2)
    s[j1] = i2
    s[j2] = i1
    return s


# 5) Busca local (best improvement)
def best_improvement_local_search(sol: np.ndarray, obj_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
    best_sol = sol.copy()
    best_val = obj_func(best_sol)

    improved = True
    while improved:
        improved = False
        current_best_sol = best_sol
        current_best_val = best_val

        for j in range(n):
            current_agent = best_sol[j]
            for i in range(m):
                if i == current_agent:
                    continue
                candidate = best_sol.copy()
                candidate[j] = i
                if is_feasible(candidate):
                    candidate_val = obj_func(candidate)
                    if candidate_val < current_best_val:
                        current_best_sol = candidate
                        current_best_val = candidate_val

        if current_best_val < best_val:
            best_sol, best_val = current_best_sol, current_best_val
            improved = True

    return best_sol, best_val


# 6) Metaheuristica VNS
def shake(sol: np.ndarray, k: int) -> np.ndarray:
    s = sol.copy()
    if k == 1:
        s = neighborhood_shift(s)
    elif k == 2:
        s = neighborhood_exchange(s)
    elif k == 3:
        s = neighborhood_swap(s)
    else:
        for _ in range(k - 2):
            s = neighborhood_shift(s)

    if not is_feasible(s):
        return sol
    return s


def VNS(obj_func: Callable[[np.ndarray], float], max_iter: int = 400, k_max: int = 3) -> Tuple[np.ndarray, float, List[float]]:
    sol = None
    while sol is None:
        sol = greedy_solution_grasp(alpha=0.3)

    best_sol, best_val = best_improvement_local_search(sol, obj_func)
    history = [best_val]

    for _ in range(max_iter):
        k = 1
        while k <= k_max:
            s_shake = shake(best_sol, k)
            s_local, val_local = best_improvement_local_search(s_shake, obj_func)
            if val_local < best_val:
                best_sol, best_val = s_local, val_local
                k = 1
            else:
                k += 1
        history.append(best_val)

    return best_sol, best_val, history


# 7) Funcoes de escalarização
def obj_func_ponderada(sol: np.ndarray, w1: float, f1_min: float, f1_max: float, f2_min: float, f2_max: float) -> float:
    val_f1 = f1(sol)
    val_f2 = f2(sol)
    if val_f2 == float("inf"):
        return float("inf")
    norm_f1 = (val_f1 - f1_min) / (f1_max - f1_min + 1e-6)
    norm_f2 = (val_f2 - f2_min) / (f2_max - f2_min + 1e-6)
    w2 = 1.0 - w1
    return (w1 * norm_f1) + (w2 * norm_f2)


def obj_func_epsilon(sol: np.ndarray, epsilon_val: float) -> float:
    val_f1 = f1(sol)
    val_f2 = f2(sol)
    if val_f1 == float("inf") or val_f2 == float("inf"):
        return float("inf")
    if val_f2 > epsilon_val:
        penalty = 1000 * (val_f2 - epsilon_val)
        return val_f1 + penalty
    return val_f1


# 8) Execucao multiobjetivo (Pw e Epsilon)
def get_nadir_points(max_iter: int = 300) -> Tuple[float, float, float, float]:
    print("  - Buscando pontos ideais/nadir com VNS mono-objetivo...")
    best_sol_f1, f1_min, _ = VNS(f1, max_iter=max_iter, k_max=3)
    f2_max = f2(best_sol_f1)
    best_sol_f2, f2_min, _ = VNS(f2, max_iter=max_iter, k_max=3)
    f1_max = f1(best_sol_f2)
    return f1_min, f1_max, f2_min, f2_max


def run_soma_ponderada(
    f1_min: float,
    f1_max: float,
    f2_min: float,
    f2_max: float,
    num_steps: int = 20,
    vns_iter: int = 200,
) -> List[Dict]:
    frontier = []
    for w1 in np.linspace(0.0, 1.0, num_steps):
        obj_func = lambda sol: obj_func_ponderada(sol, w1, f1_min, f1_max, f2_min, f2_max)
        best_sol, _, _ = VNS(obj_func, max_iter=vns_iter, k_max=3)
        frontier.append({"sol": best_sol, "f1": f1(best_sol), "f2": f2(best_sol), "method": "Pw", "param": w1})
    return frontier


def run_epsilon_restrito(
    f1_min: float,
    f1_max: float,
    f2_min: float,
    f2_max: float,
    num_steps: int = 20,
    vns_iter: int = 200,
) -> List[Dict]:
    frontier = []
    for epsilon_val in np.linspace(f2_max, f2_min, num_steps):
        obj_func = lambda sol: obj_func_epsilon(sol, epsilon_val)
        best_sol, best_val, _ = VNS(obj_func, max_iter=vns_iter, k_max=3)
        if best_val < float("inf"):
            frontier.append({"sol": best_sol, "f1": f1(best_sol), "f2": f2(best_sol), "method": "Eps", "param": epsilon_val})
    return frontier


def filter_non_dominated(solutions: List[Dict]) -> List[Dict]:
    ordered = sorted(solutions, key=lambda s: (s["f1"], s["f2"]))
    pareto = []
    for cand in ordered:
        dominated = False
        for p in pareto:
            if p["f1"] <= cand["f1"] and p["f2"] <= cand["f2"] and (p["f1"] < cand["f1"] or p["f2"] < cand["f2"]):
                dominated = True
                break
        if not dominated:
            pareto = [p for p in pareto if not (cand["f1"] <= p["f1"] and cand["f2"] <= p["f2"] and (cand["f1"] < p["f1"] or cand["f2"] < p["f2"]))]
            pareto.append(cand)
    return pareto


def select_representative(pareto: List[Dict], max_points: int = 20) -> List[Dict]:
    if len(pareto) <= max_points:
        return pareto
    pareto_sorted = sorted(pareto, key=lambda s: s["f1"])
    indices = np.linspace(0, len(pareto_sorted) - 1, max_points, dtype=int)
    return [pareto_sorted[i] for i in indices]


# 9) Atributos adicionais para a tomada de decisao
def evaluate_attributes(sol: np.ndarray, base_cost: float) -> Dict[str, float]:
    loads = compute_loads(sol)
    slack_min = float(np.min(b - loads))
    slack_mean = float(np.mean(b - loads))

    rng = np.random.default_rng(123)
    viol_sum = 0.0
    cost_var_sum = 0.0
    num_scenarios = 25
    for _ in range(num_scenarios):
        cap_factor = rng.uniform(0.9, 1.1, size=b.shape)
        cap_perturbed = b * cap_factor
        violation = np.maximum(loads - cap_perturbed, 0).sum()
        viol_sum += violation

        cost_factor = rng.uniform(0.9, 1.1, size=c.shape)
        pert_cost_matrix = c * cost_factor
        pert_cost = float(sum(pert_cost_matrix[sol[j], j] for j in range(n)))
        cost_var_sum += abs(pert_cost - base_cost)

    violation_mean = viol_sum / num_scenarios
    cost_var_mean = cost_var_sum / num_scenarios

    return {
        "slack_min": slack_min,
        "slack_mean": slack_mean,
        "violation_mean": violation_mean,
        "cost_var_mean": cost_var_mean,
    }


def build_attribute_table(candidates: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    enriched = []
    rows = []
    for idx, cand in enumerate(candidates):
        attrs = evaluate_attributes(cand["sol"], cand["f1"])
        data = {
            "id": idx,
            "f1": cand["f1"],
            "f2": cand["f2"],
            "slack_min": attrs["slack_min"],
            "cost_var": attrs["cost_var_mean"],
            "slack_mean": attrs["slack_mean"],
            "violation_mean": attrs["violation_mean"],
            "method": cand.get("method", ""),
            "param": cand.get("param", 0.0),
        }
        enriched.append({**cand, **data})
        rows.append([data["f1"], data["f2"], data["slack_min"], data["cost_var"]])
    return np.array(rows, dtype=float), enriched


# 10) Metodos de decisão (abordagem classica e TOPSIS)
def normalize_matrix(matrix: np.ndarray, criteria_types: List[str]) -> np.ndarray:
    norm = np.zeros_like(matrix, dtype=float)
    for j, ctype in enumerate(criteria_types):
        col = matrix[:, j]
        col_min, col_max = np.min(col), np.max(col)
        if abs(col_max - col_min) < 1e-9:
            norm[:, j] = 1.0
            continue
        if ctype == "benefit":
            norm[:, j] = (col - col_min) / (col_max - col_min)
        else:
            norm[:, j] = (col_max - col) / (col_max - col_min)
    return norm


def classical_weighted_sum(matrix: np.ndarray, weights: np.ndarray, criteria_types: List[str]) -> np.ndarray:
    norm = normalize_matrix(matrix, criteria_types)
    return norm.dot(weights)


def topsis(matrix: np.ndarray, weights: np.ndarray, criteria_types: List[str]) -> np.ndarray:
    # Normalizacao vetorial
    denom = np.linalg.norm(matrix, axis=0)
    denom[denom == 0] = 1.0
    norm = matrix / denom
    weighted = norm * weights

    ideal = []
    nadir = []
    for j, ctype in enumerate(criteria_types):
        if ctype == "benefit":
            ideal.append(np.max(weighted[:, j]))
            nadir.append(np.min(weighted[:, j]))
        else:
            ideal.append(np.min(weighted[:, j]))
            nadir.append(np.max(weighted[:, j]))
    ideal = np.array(ideal)
    nadir = np.array(nadir)

    dist_pos = np.linalg.norm(weighted - ideal, axis=1)
    dist_neg = np.linalg.norm(weighted - nadir, axis=1)
    score = dist_neg / (dist_pos + dist_neg + 1e-12)
    return score


# 11) Plots
def plot_frontier(candidates: List[Dict], idx_classic: int, idx_topsis: int) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "axes.facecolor": "#f7f9fc",
        "figure.facecolor": "white",
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
    })

    palette = {"Pw": "#3b82f6", "Eps": "#f97316"}
    markers = {"Pw": "o", "Eps": "s"}
    plt.figure(figsize=(10.5, 7.2))
    for cand in candidates:
        plt.scatter(
            cand["f1"],
            cand["f2"],
            s=85,
            color=palette.get(cand.get("method", ""), "#6b7280"),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            marker=markers.get(cand.get("method", ""), "o"),
        )

    plt.scatter(
        candidates[idx_classic]["f1"],
        candidates[idx_classic]["f2"],
        s=230,
        color="#0ea5e9",
        edgecolor="black",
        linewidth=1.2,
        marker="o",
        label="Escolha classica",
        zorder=4,
    )
    plt.scatter(
        candidates[idx_topsis]["f1"],
        candidates[idx_topsis]["f2"],
        s=250,
        color="#ef4444",
        edgecolor="black",
        linewidth=1.2,
        marker="X",
        label="Escolha TOPSIS",
        zorder=5,
    )

    plt.annotate("Clássica", (candidates[idx_classic]["f1"], candidates[idx_classic]["f2"]),
                 xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold", color="#0ea5e9")
    plt.annotate("TOPSIS", (candidates[idx_topsis]["f1"], candidates[idx_topsis]["f2"]),
                 xytext=(8, -14), textcoords="offset points", fontsize=11, fontweight="bold", color="#ef4444")

    plt.xlabel("f1 (custo)")
    plt.ylabel("f2 (desequilíbrio)")
    plt.title("Fronteira não-dominada para decisão")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Pw", markerfacecolor=palette["Pw"], markeredgecolor="white", markersize=11),
        plt.Line2D([0], [0], marker="s", color="w", label="Eps", markerfacecolor=palette["Eps"], markeredgecolor="white", markersize=11),
    ]
    plt.legend(handles=legend_handles + plt.gca().get_legend_handles_labels()[0], loc="best", frameon=True, fontsize=12, borderpad=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "decisao_multicriterio", "decisao_fronteira.png"), dpi=220)
    plt.close()


def plot_solution_loads(solutions: List[Dict], labels: List[str], out_path: str) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(11, 6))
    width = 0.36
    x = np.arange(m)
    colors = ["#0ea5e9", "#ef4444", "#10b981"]
    for idx, sol in enumerate(solutions):
        loads = compute_loads(sol["sol"])
        offset = (idx - (len(solutions) - 1) / 2) * width
        plt.bar(
            x + offset,
            loads,
            width=width,
            alpha=0.85,
            label=labels[idx],
            color=colors[idx % len(colors)],
            edgecolor="white",
            linewidth=0.6,
        )
    plt.xticks(x, [f"Agente {i+1}" for i in range(m)])
    plt.ylabel("Carga total")
    plt.xlabel("Agentes")
    plt.title("Cargas por agente nas soluções escolhidas")
    plt.legend()
    plt.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# 12) Main
def main():
    random.seed(42)
    np.random.seed(42)
    base_dir = os.path.join("graphs", "decisao_multicriterio")
    os.makedirs(base_dir, exist_ok=True)

    fast_mode = os.getenv("ENTREGA3_FAST", "0") == "1"
    if fast_mode:
        num_execucoes = 1
        num_pontos = 6
        nadir_iter = 100
        vns_pw_iter = 60
        vns_eps_iter = 60
        print("Executando em modo rapido (ENTREGA3_FAST=1)")
    else:
        num_execucoes = 5
        num_pontos = 20
        nadir_iter = 300
        vns_pw_iter = 200
        vns_eps_iter = 200

    all_solutions = []

    print("=== ENT3: coleta das fronteiras ===")
    t0 = time.time()
    for run in range(num_execucoes):
        print(f"\nExecucao {run + 1}/{num_execucoes}")
        f1_min, f1_max, f2_min, f2_max = get_nadir_points(max_iter=nadir_iter)
        front_pw = run_soma_ponderada(f1_min, f1_max, f2_min, f2_max, num_steps=num_pontos, vns_iter=vns_pw_iter)
        front_eps = run_epsilon_restrito(f1_min, f1_max, f2_min, f2_max, num_steps=num_pontos, vns_iter=vns_eps_iter)
        all_solutions.extend(front_pw + front_eps)
    print(f"Tempo total de coleta: {time.time() - t0:.2f}s")

    print("\n=== Construindo fronteira unificada ===")
    pareto = filter_non_dominated(all_solutions)
    pareto_repr = select_representative(pareto, max_points=20)
    print(f"Solucoes nao-dominadas consideradas: {len(pareto_repr)}")

    print("\n=== Avaliando atributos adicionais ===")
    matrix, enriched = build_attribute_table(pareto_repr)

    # Atributos usados na decisao:
    # 0) f1 (custo) -> crit. de custo
    # 1) f2 (desequilibrio) -> crit. de custo
    # 2) folga minima -> beneficio
    # 3) variacao de custo frente a ruido -> crit. de custo
    criteria_types = ["cost", "cost", "benefit", "cost"]
    weights = np.array([0.35, 0.25, 0.20, 0.20])

    classic_scores = classical_weighted_sum(matrix, weights, criteria_types)
    topsis_scores = topsis(matrix, weights, criteria_types)

    idx_classic = int(np.argmax(classic_scores))
    idx_topsis = int(np.argmax(topsis_scores))

    print("\n=== Ranking (abordagem classica) ===")
    for rank, (i, score) in enumerate(sorted(enumerate(classic_scores), key=lambda t: t[1], reverse=True), start=1):
        print(f"Rank {rank:02d} | cand {enriched[i]['id']:02d} | score {score:.4f} | f1={enriched[i]['f1']:.2f} | f2={enriched[i]['f2']:.2f}")

    print("\n=== Ranking (TOPSIS) ===")
    for rank, (i, score) in enumerate(sorted(enumerate(topsis_scores), key=lambda t: t[1], reverse=True), start=1):
        print(f"Rank {rank:02d} | cand {enriched[i]['id']:02d} | score {score:.4f} | f1={enriched[i]['f1']:.2f} | f2={enriched[i]['f2']:.2f}")

    winners = {
        "classic": enriched[idx_classic],
        "topsis": enriched[idx_topsis],
    }

    df = pd.DataFrame(
        [
            {
                "cand": e["id"],
                "f1": e["f1"],
                "f2": e["f2"],
                "slack_min": e["slack_min"],
                "slack_mean": e["slack_mean"],
                "violation_mean": e["violation_mean"],
                "cost_var": e["cost_var"],
                "method": e["method"],
                "param": e["param"],
                "score_classic": classic_scores[i],
                "score_topsis": topsis_scores[i],
            }
            for i, e in enumerate(enriched)
        ]
    )
    resumo_path = os.path.join(base_dir, "decisao_resumo.csv")
    df.to_csv(resumo_path, index=False)
    print(f"\nResumo salvo em {resumo_path}")

    cargas_path = os.path.join(base_dir, "decisao_cargas.png")
    plot_frontier(enriched, idx_classic, idx_topsis)
    plot_solution_loads([winners["classic"], winners["topsis"]], ["Classica", "TOPSIS"], cargas_path)
    print(f"Graficos salvos em {os.path.join(base_dir, 'decisao_fronteira.png')} e {cargas_path}")

    print("\nSolucoes escolhidas:")
    for name, sol in winners.items():
        print(f"- {name}: f1={sol['f1']:.2f}, f2={sol['f2']:.2f}, slack_min={sol['slack_min']:.2f}, cost_var={sol['cost_var']:.2f}")


if __name__ == "__main__":
    main()
