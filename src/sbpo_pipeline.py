import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog


GRAPH_DIR_ROOT = os.path.join("graphs", "sbpo_fitradeoff")
GRAPH_DIR = GRAPH_DIR_ROOT
CRITERIA_SPECS = [
    ("f1", "Custo total", "cost"),
    ("f2", "Desequilibrio de utilizacao", "cost"),
    ("slack_min", "Folga minima", "benefit"),
    ("cost_var", "Variacao media de custo", "cost"),
]
REFERENCE_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.20], dtype=float)
PROGRESS_REPORT_INTERVAL = 30.0


@dataclass
class TradeoffQuestion:
    more_important: str
    less_important: str
    partial_value: float
    preferred_side: str
    constraint: str
    prompt: str


def emit(message: str) -> None:
    print(message, flush=True)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def format_value(value: float) -> str:
    if np.isfinite(value):
        return f"{value:.4f}"
    return "inf"


def load_csv_instance(base_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    costs = pd.read_csv(os.path.join(base_dir, "custos.csv"), header=None).values.astype(float)
    resources = pd.read_csv(os.path.join(base_dir, "recursos.csv"), header=None).values.astype(float)
    capacities = pd.read_csv(os.path.join(base_dir, "capacidades.csv"), header=None).values.flatten().astype(float)
    return costs, resources, capacities


def load_orlib_instance(orlib_file: str, problem_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(orlib_file, "r", encoding="utf-8") as handle:
        tokens = [int(token) for token in handle.read().split()]

    total_instances = tokens[0]
    if problem_index < 1 or problem_index > total_instances:
        raise ValueError(f"problem_index deve estar entre 1 e {total_instances}.")

    cursor = 1
    for current_index in range(1, total_instances + 1):
        num_agents = tokens[cursor]
        num_tasks = tokens[cursor + 1]
        cursor += 2

        size = num_agents * num_tasks
        costs = np.array(tokens[cursor: cursor + size], dtype=float).reshape(num_agents, num_tasks)
        cursor += size
        resources = np.array(tokens[cursor: cursor + size], dtype=float).reshape(num_agents, num_tasks)
        cursor += size
        capacities = np.array(tokens[cursor: cursor + num_agents], dtype=float)
        cursor += num_agents

        if current_index == problem_index:
            return costs, resources, capacities

    raise RuntimeError("Instancia OR-Library nao encontrada.")


def set_instance(costs: np.ndarray, resources: np.ndarray, capacities: np.ndarray) -> None:
    global COSTS, RESOURCES, CAPACITIES, M_AGENTS, N_TASKS
    COSTS = costs
    RESOURCES = resources
    CAPACITIES = capacities
    M_AGENTS, N_TASKS = costs.shape


try:
    set_instance(*load_csv_instance())
except FileNotFoundError:
    print("Arquivos de dados nao encontrados. Gerando instancia aleatoria de teste.")
    random_costs = np.random.rand(5, 50) * 10.0
    random_resources = np.random.rand(5, 50) * 5.0
    random_capacities = np.full(5, 100.0)
    set_instance(random_costs, random_resources, random_capacities)


def f1(solution: np.ndarray) -> float:
    return float(sum(COSTS[solution[j], j] for j in range(N_TASKS)))


def compute_utilization_rates(loads: np.ndarray) -> np.ndarray:
    utilization = np.zeros_like(loads, dtype=float)
    positive_capacity = CAPACITIES > 0
    utilization[positive_capacity] = loads[positive_capacity] / CAPACITIES[positive_capacity]
    return utilization


def f2(solution: np.ndarray) -> float:
    loads = compute_loads(solution)
    if np.any(loads > CAPACITIES):
        return float("inf")
    utilization = compute_utilization_rates(loads)
    return float(np.max(utilization) - np.min(utilization))


def compute_loads(solution: np.ndarray) -> np.ndarray:
    loads = np.zeros(M_AGENTS)
    for j in range(N_TASKS):
        loads[solution[j]] += RESOURCES[solution[j], j]
    return loads


def is_feasible(solution: np.ndarray) -> bool:
    return bool(np.all(compute_loads(solution) <= CAPACITIES))


def greedy_solution_grasp(alpha: float = 0.30) -> np.ndarray | None:
    solution = -np.ones(N_TASKS, dtype=int)
    loads = np.zeros(M_AGENTS)
    tasks = list(range(N_TASKS))
    random.shuffle(tasks)

    for task in tasks:
        feasible_agents = []
        for agent in range(M_AGENTS):
            if loads[agent] + RESOURCES[agent, task] <= CAPACITIES[agent]:
                feasible_agents.append((COSTS[agent, task], agent))

        if not feasible_agents:
            return None

        feasible_agents.sort()
        best_cost = feasible_agents[0][0]
        max_rcl_cost = best_cost + alpha * (feasible_agents[-1][0] - best_cost)
        rcl = [agent for cost, agent in feasible_agents if cost <= max_rcl_cost]
        if not rcl:
            rcl = [feasible_agents[0][1]]

        chosen = random.choice(rcl)
        solution[task] = chosen
        loads[chosen] += RESOURCES[chosen, task]

    return solution


def neighborhood_shift(solution: np.ndarray) -> np.ndarray:
    candidate = solution.copy()
    task = random.randrange(N_TASKS)
    current_agent = candidate[task]
    new_agent = random.randint(0, M_AGENTS - 1)
    while new_agent == current_agent:
        new_agent = random.randint(0, M_AGENTS - 1)
    candidate[task] = new_agent
    return candidate


def neighborhood_exchange(solution: np.ndarray) -> np.ndarray:
    candidate = solution.copy()
    task_a, task_b = random.sample(range(N_TASKS), 2)
    candidate[task_a], candidate[task_b] = candidate[task_b], candidate[task_a]
    return candidate


def neighborhood_swap(solution: np.ndarray) -> np.ndarray:
    candidate = solution.copy()
    active_agents = list(set(candidate))
    if len(active_agents) < 2:
        return candidate

    agent_a, agent_b = random.sample(active_agents, 2)
    tasks_a = np.where(candidate == agent_a)[0]
    tasks_b = np.where(candidate == agent_b)[0]
    if len(tasks_a) == 0 or len(tasks_b) == 0:
        return candidate

    task_a = random.choice(tasks_a)
    task_b = random.choice(tasks_b)
    candidate[task_a] = agent_b
    candidate[task_b] = agent_a
    return candidate


def best_improvement_local_search(
    solution: np.ndarray,
    objective: Callable[[np.ndarray], float],
    progress_label: str | None = None,
    report_interval: float = PROGRESS_REPORT_INTERVAL,
) -> Tuple[np.ndarray, float]:
    best_solution = solution.copy()
    best_value = objective(best_solution)
    search_start = time.time()
    last_report = search_start
    pass_count = 0

    improved = True
    while improved:
        pass_count += 1
        improved = False
        move_solution = best_solution
        move_value = best_value
        pass_start = time.time()

        for task in range(N_TASKS):
            current_agent = best_solution[task]
            for agent in range(M_AGENTS):
                if agent == current_agent:
                    continue
                candidate = best_solution.copy()
                candidate[task] = agent
                if not is_feasible(candidate):
                    continue
                candidate_value = objective(candidate)
                if candidate_value < move_value:
                    move_solution = candidate
                    move_value = candidate_value

            now = time.time()
            if progress_label and now - last_report >= report_interval:
                completed = task + 1
                elapsed_pass = now - pass_start
                eta_pass = (elapsed_pass / completed) * (N_TASKS - completed) if completed > 0 else 0.0
                emit(
                    f"[{progress_label}] busca local p{pass_count} | "
                    f"tarefa {completed}/{N_TASKS} ({completed / N_TASKS:.0%}) | "
                    f"melhor={format_value(move_value)} | "
                    f"tempo_pass={format_seconds(elapsed_pass)} | "
                    f"eta_pass={format_seconds(eta_pass)}"
                )
                last_report = now

        if move_value < best_value:
            best_solution = move_solution
            best_value = move_value
            improved = True

    total_search_time = time.time() - search_start
    if progress_label and total_search_time >= report_interval:
        emit(
            f"[{progress_label}] busca local concluida | "
            f"passes={pass_count} | melhor={format_value(best_value)} | "
            f"tempo={format_seconds(total_search_time)}"
        )

    return best_solution, best_value


def shake(solution: np.ndarray, k: int) -> np.ndarray:
    candidate = solution.copy()
    if k == 1:
        candidate = neighborhood_shift(candidate)
    elif k == 2:
        candidate = neighborhood_exchange(candidate)
    elif k == 3:
        candidate = neighborhood_swap(candidate)
    else:
        for _ in range(k - 2):
            candidate = neighborhood_shift(candidate)

    if not is_feasible(candidate):
        return solution
    return candidate


def vns(
    objective: Callable[[np.ndarray], float],
    max_iter: int = 250,
    k_max: int = 3,
    progress_label: str | None = None,
    report_interval: float = PROGRESS_REPORT_INTERVAL,
) -> Tuple[np.ndarray, float, List[float]]:
    solution = None
    while solution is None:
        solution = greedy_solution_grasp(alpha=0.30)

    vns_start = time.time()
    last_report = vns_start
    progress_every = max(1, max_iter // 5)

    if progress_label:
        emit(f"[{progress_label}] VNS iniciada | max_iter={max_iter} | k_max={k_max}")

    best_solution, best_value = best_improvement_local_search(
        solution,
        objective,
        progress_label=f"{progress_label} | inicial" if progress_label else None,
        report_interval=report_interval,
    )
    history = [best_value]

    for iteration in range(1, max_iter + 1):
        k = 1
        while k <= k_max:
            shaken = shake(best_solution, k)
            local_solution, local_value = best_improvement_local_search(
                shaken,
                objective,
                progress_label=f"{progress_label} | iter {iteration}/{max_iter} k{k}" if progress_label else None,
                report_interval=report_interval,
            )
            if local_value < best_value:
                best_solution = local_solution
                best_value = local_value
                k = 1
            else:
                k += 1
        history.append(best_value)

        now = time.time()
        should_report = iteration == 1 or iteration == max_iter or iteration % progress_every == 0 or now - last_report >= report_interval
        if progress_label and should_report:
            elapsed = now - vns_start
            eta = (elapsed / iteration) * (max_iter - iteration) if iteration > 0 else 0.0
            emit(
                f"[{progress_label}] iter {iteration}/{max_iter} ({iteration / max_iter:.0%}) | "
                f"melhor={format_value(best_value)} | "
                f"tempo={format_seconds(elapsed)} | "
                f"eta={format_seconds(eta)}"
            )
            last_report = now

    if progress_label:
        emit(f"[{progress_label}] VNS concluida | melhor={format_value(best_value)} | tempo={format_seconds(time.time() - vns_start)}")

    return best_solution, best_value, history


def obj_func_epsilon(solution: np.ndarray, epsilon_value: float) -> float:
    value_f1 = f1(solution)
    value_f2 = f2(solution)
    if value_f1 == float("inf") or value_f2 == float("inf"):
        return float("inf")
    if value_f2 > epsilon_value:
        return value_f1 + 1000.0 * (value_f2 - epsilon_value)
    return value_f1


def get_nadir_points(max_iter: int = 250) -> Tuple[float, float, float, float]:
    emit(f"[referencia] Calculando ponto de referencia para f1 | max_iter={max_iter}")
    best_solution_f1, f1_min, _ = vns(f1, max_iter=max_iter, k_max=3, progress_label="referencia f1")
    f2_max = f2(best_solution_f1)
    emit(f"[referencia] Calculando ponto de referencia para f2 | max_iter={max_iter}")
    best_solution_f2, f2_min, _ = vns(f2, max_iter=max_iter, k_max=3, progress_label="referencia f2")
    f1_max = f1(best_solution_f2)
    return f1_min, f1_max, f2_min, f2_max


def run_epsilon_restrito(
    f2_min: float,
    f2_max: float,
    num_steps: int,
    vns_iter: int,
    run_id: int,
) -> List[Dict]:
    frontier = []
    epsilon_values = np.linspace(f2_max, f2_min, num_steps)
    run_start = time.time()
    for step_idx, epsilon_value in enumerate(epsilon_values, start=1):
        elapsed = time.time() - run_start
        eta = (elapsed / (step_idx - 1)) * (num_steps - step_idx + 1) if step_idx > 1 else 0.0
        emit(
            f"[epsilon run {run_id + 1}] ponto {step_idx}/{num_steps} | "
            f"epsilon={epsilon_value:.6f} | "
            f"tempo={format_seconds(elapsed)} | "
            f"eta={format_seconds(eta)}"
        )
        objective = lambda sol: obj_func_epsilon(sol, epsilon_value)
        best_solution, best_value, _ = vns(
            objective,
            max_iter=vns_iter,
            k_max=3,
            progress_label=f"epsilon run {run_id + 1} ponto {step_idx}/{num_steps}",
        )
        if best_value == float("inf"):
            continue
        frontier.append(
            {
                "run": run_id,
                "sol": best_solution,
                "f1": f1(best_solution),
                "f2": f2(best_solution),
                "epsilon": float(epsilon_value),
            }
        )
        emit(
            f"[epsilon run {run_id + 1}] ponto {step_idx}/{num_steps} concluido | "
            f"f1={f1(best_solution):.2f} | f2={f2(best_solution):.4f}"
        )
    return frontier


def unique_solutions(solutions: Sequence[Dict]) -> List[Dict]:
    unique = []
    seen = set()
    for solution in solutions:
        key = (
            round(float(solution["f1"]), 8),
            round(float(solution["f2"]), 8),
            tuple(int(x) for x in solution["sol"].tolist()),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(solution)
    return unique


def filter_non_dominated(solutions: Sequence[Dict]) -> List[Dict]:
    ordered = sorted(unique_solutions(solutions), key=lambda item: (item["f1"], item["f2"]))
    pareto = []
    for candidate in ordered:
        dominated = False
        for incumbent in pareto:
            better_f1 = incumbent["f1"] <= candidate["f1"]
            better_f2 = incumbent["f2"] <= candidate["f2"]
            strict = incumbent["f1"] < candidate["f1"] or incumbent["f2"] < candidate["f2"]
            if better_f1 and better_f2 and strict:
                dominated = True
                break
        if dominated:
            continue
        pareto = [
            incumbent
            for incumbent in pareto
            if not (
                candidate["f1"] <= incumbent["f1"]
                and candidate["f2"] <= incumbent["f2"]
                and (candidate["f1"] < incumbent["f1"] or candidate["f2"] < incumbent["f2"])
            )
        ]
        pareto.append(candidate)
    return pareto


def select_representative(solutions: Sequence[Dict], max_points: int = 12) -> List[Dict]:
    ordered = sorted(solutions, key=lambda item: (item["f1"], item["f2"]))
    if len(ordered) <= max_points:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, max_points, dtype=int)
    return [ordered[index] for index in indices]


def evaluate_attributes(solution: np.ndarray, base_cost: float) -> Dict[str, float]:
    loads = compute_loads(solution)
    utilization = compute_utilization_rates(loads)
    slack = CAPACITIES - loads
    slack_min = float(np.min(slack))
    slack_mean = float(np.mean(slack))

    rng = np.random.default_rng(123)
    violation_sum = 0.0
    cost_var_sum = 0.0
    num_scenarios = 25

    for _ in range(num_scenarios):
        cap_factor = rng.uniform(0.90, 1.10, size=CAPACITIES.shape)
        cap_perturbed = CAPACITIES * cap_factor
        violation_sum += float(np.maximum(loads - cap_perturbed, 0.0).sum())

        cost_factor = rng.uniform(0.90, 1.10, size=COSTS.shape)
        perturbed_costs = COSTS * cost_factor
        perturbed_cost = float(sum(perturbed_costs[solution[j], j] for j in range(N_TASKS)))
        cost_var_sum += abs(perturbed_cost - base_cost)

    return {
        "slack_min": slack_min,
        "slack_mean": slack_mean,
        "util_min": float(np.min(utilization)),
        "util_max": float(np.max(utilization)),
        "violation_mean": violation_sum / num_scenarios,
        "cost_var": cost_var_sum / num_scenarios,
    }


def build_candidate_frame(candidates: Sequence[Dict]) -> pd.DataFrame:
    rows = []
    for candidate_id, candidate in enumerate(candidates):
        attributes = evaluate_attributes(candidate["sol"], candidate["f1"])
        rows.append(
            {
                "cand_id": candidate_id,
                "run": candidate.get("run", -1),
                "epsilon": candidate.get("epsilon", 0.0),
                "f1": float(candidate["f1"]),
                "f2": float(candidate["f2"]),
                "slack_min": attributes["slack_min"],
                "slack_mean": attributes["slack_mean"],
                "util_min": attributes["util_min"],
                "util_max": attributes["util_max"],
                "violation_mean": attributes["violation_mean"],
                "cost_var": attributes["cost_var"],
                "sol": candidate["sol"],
            }
        )
    return pd.DataFrame(rows)


def normalize_to_benefit_scale(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame({"cand_id": frame["cand_id"].astype(int)})
    for name, _, kind in CRITERIA_SPECS:
        column = frame[name].to_numpy(dtype=float)
        col_min = float(np.min(column))
        col_max = float(np.max(column))
        if abs(col_max - col_min) < 1e-9:
            normalized[name] = np.ones_like(column)
        elif kind == "benefit":
            normalized[name] = (column - col_min) / (col_max - col_min)
        else:
            normalized[name] = (col_max - column) / (col_max - col_min)
    return normalized


def build_tradeoff_questions(
    criterion_names: Sequence[str],
    reference_weights: np.ndarray,
    step: float = 0.10,
) -> List[TradeoffQuestion]:
    questions = []
    for index in range(len(criterion_names) - 1):
        more_important = criterion_names[index]
        less_important = criterion_names[index + 1]
        ratio = reference_weights[index + 1] / reference_weights[index]

        lower = math.floor((ratio - 1e-9) / step) * step
        if lower >= step:
            prompt = (
                f"{lower:.0%} de melhoria em {more_important} versus "
                f"100% de melhoria em {less_important}"
            )
            questions.append(
                TradeoffQuestion(
                    more_important=more_important,
                    less_important=less_important,
                    partial_value=round(lower, 2),
                    preferred_side="less_important",
                    constraint=f"w_{less_important} >= {lower:.2f} * w_{more_important}",
                    prompt=prompt,
                )
            )

        upper = math.ceil((ratio + 1e-9) / step) * step
        if upper <= 1.0 and upper - ratio > 1e-9:
            prompt = (
                f"{upper:.0%} de melhoria em {more_important} versus "
                f"100% de melhoria em {less_important}"
            )
            questions.append(
                TradeoffQuestion(
                    more_important=more_important,
                    less_important=less_important,
                    partial_value=round(upper, 2),
                    preferred_side="more_important",
                    constraint=f"w_{less_important} <= {upper:.2f} * w_{more_important}",
                    prompt=prompt,
                )
            )

    return questions


def build_weight_constraints(
    criterion_names: Sequence[str],
    questions: Sequence[TradeoffQuestion],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    criterion_index = {name: index for index, name in enumerate(criterion_names)}
    num_criteria = len(criterion_names)

    a_ub = []
    b_ub = []
    for index in range(num_criteria - 1):
        row = np.zeros(num_criteria)
        row[index + 1] = 1.0
        row[index] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    for question in questions:
        row = np.zeros(num_criteria)
        more_idx = criterion_index[question.more_important]
        less_idx = criterion_index[question.less_important]
        if question.preferred_side == "less_important":
            row[more_idx] = question.partial_value
            row[less_idx] = -1.0
        else:
            row[more_idx] = -question.partial_value
            row[less_idx] = 1.0
        a_ub.append(row)
        b_ub.append(0.0)

    a_eq = np.ones((1, num_criteria))
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * num_criteria
    return np.array(a_ub), np.array(b_ub), a_eq, b_eq, bounds


def solve_weight_lp(
    objective: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"FITradeoff LP inviavel: {result.message}")
    return result.x


def compute_weight_intervals(
    criterion_names: Sequence[str],
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> pd.DataFrame:
    rows = []
    num_criteria = len(criterion_names)
    for index, name in enumerate(criterion_names):
        objective = np.zeros(num_criteria)
        objective[index] = 1.0
        minimum = solve_weight_lp(objective, a_ub, b_ub, a_eq, b_eq, bounds)[index]
        maximum = solve_weight_lp(-objective, a_ub, b_ub, a_eq, b_eq, bounds)[index]
        rows.append(
            {
                "criterion": name,
                "min_weight": float(minimum),
                "max_weight": float(maximum),
            }
        )
    return pd.DataFrame(rows)


def potentially_optimal_alternatives(
    value_matrix: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> Dict[int, Dict[str, np.ndarray | float]]:
    num_alternatives, num_criteria = value_matrix.shape
    if num_alternatives == 1:
        witness_weights = solve_weight_lp(
            np.zeros(num_criteria),
            a_ub,
            b_ub,
            a_eq,
            b_eq,
            bounds,
        )
        return {0: {"margin": float("inf"), "weights": witness_weights}}

    potentials = {}

    for alt_index in range(num_alternatives):
        objective = np.zeros(num_criteria + 1)
        objective[-1] = -1.0

        lp_a_ub = [np.append(row, 0.0) for row in a_ub]
        lp_b_ub = list(b_ub)

        for other_index in range(num_alternatives):
            if other_index == alt_index:
                continue
            row = np.append(-(value_matrix[alt_index] - value_matrix[other_index]), 1.0)
            lp_a_ub.append(row)
            lp_b_ub.append(0.0)

        lp_a_eq = np.array([np.append(a_eq[0], 0.0)])
        lp_b_eq = b_eq.copy()
        lp_bounds = list(bounds) + [(None, None)]

        result = linprog(
            objective,
            A_ub=np.array(lp_a_ub),
            b_ub=np.array(lp_b_ub),
            A_eq=lp_a_eq,
            b_eq=lp_b_eq,
            bounds=lp_bounds,
            method="highs",
        )
        if not result.success:
            continue
        maximin_margin = float(result.x[-1])
        if maximin_margin >= -1e-8:
            potentials[alt_index] = {
                "margin": maximin_margin,
                "weights": result.x[:-1],
            }

    return potentials


def plot_frontier(all_frontier: pd.DataFrame, candidates: pd.DataFrame, chosen_id: int) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10.5, 7.0))
    plt.scatter(
        all_frontier["f1"],
        all_frontier["f2"],
        s=28,
        color="#94a3b8",
        alpha=0.45,
        label="Pontos epsilon-restrito",
    )
    plt.scatter(
        candidates["f1"],
        candidates["f2"],
        s=95,
        color="#2563eb",
        edgecolor="white",
        linewidth=0.8,
        label="Alternativas avaliadas",
    )
    chosen = candidates.loc[candidates["cand_id"] == chosen_id].iloc[0]
    plt.scatter(
        [chosen["f1"]],
        [chosen["f2"]],
        s=260,
        color="#ef4444",
        edgecolor="black",
        linewidth=1.1,
        marker="X",
        label="Escolha FITradeoff",
        zorder=5,
    )
    plt.annotate(
        f"FITradeoff (cand {chosen_id})",
        (chosen["f1"], chosen["f2"]),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="#b91c1c",
    )
    plt.xlabel("f1 (custo)")
    plt.ylabel("f2 (desequilibrio relativo de utilizacao)")
    plt.title("Fronteira epsilon-restrito com escolha FITradeoff")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "fitradeoff_fronteira.png"), dpi=220)
    plt.close()


def plot_solution_loads(chosen_row: pd.Series) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    loads = compute_loads(chosen_row["sol"])
    plt.figure(figsize=(10.0, 5.5))
    plt.bar(
        np.arange(M_AGENTS),
        loads,
        color="#0f766e",
        edgecolor="white",
        linewidth=0.8,
    )
    plt.xticks(np.arange(M_AGENTS), [f"Agente {idx + 1}" for idx in range(M_AGENTS)])
    plt.ylabel("Carga total")
    plt.xlabel("Agentes")
    plt.title("Distribuicao de carga da alternativa escolhida")
    plt.grid(axis="y", alpha=0.30)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "fitradeoff_cargas.png"), dpi=220)
    plt.close()


def plot_weight_intervals(intervals: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    labels = intervals["criterion"].tolist()
    minimums = intervals["min_weight"].to_numpy()
    maximums = intervals["max_weight"].to_numpy()
    spans = maximums - minimums

    plt.figure(figsize=(9.5, 5.5))
    plt.barh(labels, spans, left=minimums, color="#38bdf8", edgecolor="white", linewidth=0.8)
    plt.scatter(minimums, labels, color="#0f172a", s=40, label="limite inferior")
    plt.scatter(maximums, labels, color="#ef4444", s=40, label="limite superior")
    plt.xlabel("Intervalo de pesos viaveis")
    plt.title("Espaco de pesos apos a sessao FITradeoff")
    plt.xlim(0.0, max(0.45, float(maximums.max()) + 0.05))
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "fitradeoff_intervalos_pesos.png"), dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline SBPO com GRASP/VNS, epsilon-restrito e FITradeoff.")
    parser.add_argument("--fast", action="store_true", help="Executa uma configuracao mais curta para validacao.")
    parser.add_argument("--runs", type=int, default=None, help="Numero de execucoes epsilon-restrito.")
    parser.add_argument("--points", type=int, default=None, help="Numero de valores de epsilon por execucao.")
    parser.add_argument("--max-candidates", type=int, default=12, help="Numero maximo de alternativas para o FITradeoff.")
    parser.add_argument("--data-dir", type=str, default="data", help="Diretorio com custos.csv, recursos.csv e capacidades.csv.")
    parser.add_argument("--orlib-file", type=str, default=None, help="Arquivo OR-Library do GAP em formato texto.")
    parser.add_argument("--problem-index", type=int, default=1, help="Indice da instancia dentro do arquivo OR-Library.")
    parser.add_argument("--dataset-label", type=str, default=None, help="Rotulo para a pasta de saida.")
    return parser.parse_args()


def main() -> None:
    global GRAPH_DIR
    args = parse_args()
    random.seed(42)
    np.random.seed(42)

    if args.orlib_file:
        costs, resources, capacities = load_orlib_instance(args.orlib_file, args.problem_index)
        set_instance(costs, resources, capacities)
        dataset_label = args.dataset_label or f"{os.path.splitext(os.path.basename(args.orlib_file))[0]}_p{args.problem_index}"
        dataset_source = f"OR-Library: {args.orlib_file} (instancia {args.problem_index})"
    else:
        set_instance(*load_csv_instance(args.data_dir))
        default_label = "instancia_local" if os.path.normpath(args.data_dir) == os.path.normpath("data") else os.path.basename(os.path.normpath(args.data_dir))
        dataset_label = args.dataset_label or default_label
        dataset_source = f"CSV local em {args.data_dir}"

    GRAPH_DIR = os.path.join(GRAPH_DIR_ROOT, dataset_label)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    fast_mode = args.fast or os.getenv("SBPO_FAST", "0") == "1"
    if fast_mode:
        num_runs = args.runs or 1
        num_points = args.points or 6
        nadir_iter = 80
        vns_eps_iter = 50
    else:
        num_runs = args.runs or 4
        num_points = args.points or 18
        nadir_iter = 250
        vns_eps_iter = 180

    emit("=== SBPO pipeline: GRASP/VNS + epsilon-restrito + FITradeoff ===")
    emit(f"Dataset: {dataset_label}")
    emit(f"Origem: {dataset_source}")
    emit(f"Dimensao: {M_AGENTS} agentes x {N_TASKS} tarefas")
    emit(f"Modo rapido: {fast_mode}")
    emit(f"Execucoes epsilon-restrito: {num_runs}")
    emit(f"Pontos por execucao: {num_points}")

    start_time = time.time()
    f1_min, f1_max, f2_min, f2_max = get_nadir_points(max_iter=nadir_iter)
    emit(f"Pontos de referencia | f1_min={f1_min:.2f} | f1_max={f1_max:.2f} | f2_min={f2_min:.2f} | f2_max={f2_max:.2f}")

    all_solutions = []
    for run_id in range(num_runs):
        frontier = run_epsilon_restrito(
            f2_min=f2_min,
            f2_max=f2_max,
            num_steps=num_points,
            vns_iter=vns_eps_iter,
            run_id=run_id,
        )
        emit(f"Execucao {run_id + 1}/{num_runs}: {len(frontier)} pontos gerados")
        all_solutions.extend(frontier)

    all_frontier = pd.DataFrame(
        [
            {"run": item["run"], "epsilon": item["epsilon"], "f1": item["f1"], "f2": item["f2"]}
            for item in all_solutions
        ]
    )
    all_frontier.to_csv(os.path.join(GRAPH_DIR, "epsilon_frontier_all.csv"), index=False)

    pareto = filter_non_dominated(all_solutions)
    representative = select_representative(pareto, max_points=args.max_candidates)
    candidate_frame = build_candidate_frame(representative).sort_values(["f1", "f2"]).reset_index(drop=True)
    candidate_frame["cand_id"] = np.arange(len(candidate_frame))
    normalized = normalize_to_benefit_scale(candidate_frame)

    criterion_names = [name for name, _, _ in CRITERIA_SPECS]
    questions = build_tradeoff_questions(criterion_names, REFERENCE_WEIGHTS)
    a_ub, b_ub, a_eq, b_eq, bounds = build_weight_constraints(criterion_names, questions)
    weight_intervals = compute_weight_intervals(criterion_names, a_ub, b_ub, a_eq, b_eq, bounds)
    value_matrix = normalized[criterion_names].to_numpy(dtype=float)
    potentials = potentially_optimal_alternatives(value_matrix, a_ub, b_ub, a_eq, b_eq, bounds)

    if not potentials:
        raise RuntimeError("Nenhuma alternativa potencialmente otima foi encontrada pelo FITradeoff.")

    chosen_id = min(
        potentials,
        key=lambda cand_id: (-float(potentials[cand_id]["margin"]), float(candidate_frame.loc[cand_id, "f1"])),
    )

    candidate_frame["potentially_optimal"] = candidate_frame["cand_id"].isin(potentials.keys())
    candidate_frame["chosen"] = candidate_frame["cand_id"] == chosen_id
    candidate_frame["maximin_margin"] = candidate_frame["cand_id"].map(
        lambda cand_id: float(potentials.get(int(cand_id), {}).get("margin", np.nan))
    )
    for name in criterion_names:
        candidate_frame[f"value_{name}"] = normalized[name]

    chosen_weights = np.asarray(potentials[chosen_id]["weights"], dtype=float)
    witness_frame = pd.DataFrame(
        {
            "criterion": criterion_names,
            "witness_weight": chosen_weights,
        }
    )

    questions_frame = pd.DataFrame(
        [
            {
                "question_id": index + 1,
                "more_important": question.more_important,
                "less_important": question.less_important,
                "partial_value": question.partial_value,
                "preferred_side": question.preferred_side,
                "constraint": question.constraint,
                "prompt": question.prompt,
            }
            for index, question in enumerate(questions)
        ]
    )

    candidate_export = candidate_frame.drop(columns=["sol"]).copy()
    candidate_export.to_csv(os.path.join(GRAPH_DIR, "fitradeoff_candidates.csv"), index=False)
    questions_frame.to_csv(os.path.join(GRAPH_DIR, "fitradeoff_questions.csv"), index=False)
    weight_intervals.to_csv(os.path.join(GRAPH_DIR, "fitradeoff_weights.csv"), index=False)
    witness_frame.to_csv(os.path.join(GRAPH_DIR, "fitradeoff_witness_weights.csv"), index=False)

    plot_frontier(all_frontier, candidate_frame, chosen_id)
    plot_solution_loads(candidate_frame.loc[candidate_frame["cand_id"] == chosen_id].iloc[0])
    plot_weight_intervals(weight_intervals)

    chosen_row = candidate_frame.loc[candidate_frame["cand_id"] == chosen_id].iloc[0]
    emit(f"Pareto agregado: {len(pareto)} solucoes nao-dominadas")
    emit(f"Alternativas FITradeoff: {len(candidate_frame)}")
    emit(f"Alternativas potencialmente otimas: {sorted(potentials.keys())}")
    emit(f"Escolha FITradeoff: cand {chosen_id}")
    emit(
        "Resultado escolhido | "
        f"f1={chosen_row['f1']:.2f} | "
        f"f2={chosen_row['f2']:.2f} | "
        f"util_min={chosen_row['util_min']:.3f} | "
        f"util_max={chosen_row['util_max']:.3f} | "
        f"slack_min={chosen_row['slack_min']:.2f} | "
        f"cost_var={chosen_row['cost_var']:.2f}"
    )
    emit(f"Tempo total: {time.time() - start_time:.2f}s")
    emit(f"Saidas salvas em {GRAPH_DIR}")


if __name__ == "__main__":
    main()
