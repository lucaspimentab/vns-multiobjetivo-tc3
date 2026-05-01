import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class BalancedSubsetMetadata:
    dataset_name: str
    source_type: str
    derived_from_dataset: str
    source_metadata_path: str
    num_agents: int
    num_tasks: int
    allocation_mode: str
    capacity_scale: float
    max_count_ratio: float
    min_agents: int
    exact_agents: int
    target_total_tasks: int
    min_tasks_per_agent: int
    max_tasks_per_agent: int
    tasks_per_agent: int
    keep_all_tasks: bool
    random_seed: int
    selected_bases: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constroi uma instancia GAP derivada de dados reais a partir de um subconjunto mais balanceado."
    )
    parser.add_argument("--source-dir", default="data/tlc_fhv_real_large", help="Diretorio da instancia real de origem.")
    parser.add_argument("--output-dir", default="data/tlc_fhv_real_balanced", help="Diretorio de saida da nova instancia.")
    parser.add_argument(
        "--max-count-ratio",
        type=float,
        default=2.0,
        help="Razao maxima entre a maior e a menor contagem amostral entre as bases selecionadas.",
    )
    parser.add_argument(
        "--min-agents",
        type=int,
        default=6,
        help="Numero minimo de agentes no subconjunto balanceado.",
    )
    parser.add_argument(
        "--exact-agents",
        type=int,
        default=0,
        help="Numero exato de agentes desejado. Se 0, usa o maior subconjunto viavel com pelo menos --min-agents.",
    )
    parser.add_argument(
        "--target-total-tasks",
        type=int,
        default=0,
        help="Numero total de tarefas desejado com alocacao controlada por base.",
    )
    parser.add_argument(
        "--min-tasks-per-agent",
        type=int,
        default=0,
        help="Cota minima por agente quando --target-total-tasks for usado.",
    )
    parser.add_argument(
        "--max-tasks-per-agent",
        type=int,
        default=0,
        help="Cota maxima por agente quando --target-total-tasks for usado.",
    )
    parser.add_argument(
        "--tasks-per-agent",
        type=int,
        default=0,
        help="Numero de tarefas por agente. Se 0, usa a menor contagem disponivel no subconjunto selecionado.",
    )
    parser.add_argument(
        "--keep-all-tasks",
        action="store_true",
        help="Mantem todas as tarefas das bases selecionadas, em vez de igualar o numero por base.",
    )
    parser.add_argument("--capacity-scale", type=float, default=0.0, help="Folga multiplicativa das capacidades. Se 0, reutiliza a da instancia de origem.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria.")
    return parser.parse_args()


def load_source_metadata(source_dir: str) -> dict:
    metadata_path = os.path.join(source_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_balanced_bases(
    agents: pd.DataFrame,
    max_count_ratio: float,
    min_agents: int,
    exact_agents: int = 0,
) -> pd.DataFrame:
    ranked = agents.sort_values(["sample_trip_count", "dispatching_base_num"], ascending=[True, True]).reset_index(drop=True)
    best = None
    best_key = None

    for start in range(len(ranked)):
        min_count = int(ranked.loc[start, "sample_trip_count"])
        if min_count <= 0:
            continue
        min_end = start + (exact_agents if exact_agents > 0 else min_agents) - 1
        if min_end >= len(ranked):
            continue
        max_end = min_end if exact_agents > 0 else len(ranked) - 1
        for end in range(min_end, max_end + 1):
            subset = ranked.iloc[start : end + 1].copy()
            max_count = int(subset["sample_trip_count"].max())
            ratio = max_count / float(min_count)
            if ratio > max_count_ratio:
                break
            key = (len(subset), min_count, int(subset["sample_trip_count"].sum()))
            if best is None or key > best_key:
                best = subset
                best_key = key

    if best is None:
        raise RuntimeError(
            "Nao foi possivel encontrar um subconjunto balanceado. Tente aumentar --max-count-ratio ou reduzir --min-agents."
        )

    return best.sort_values("dispatching_base_num").reset_index(drop=True)


def allocate_controlled_quotas(
    source_counts: pd.Series,
    available_counts: dict,
    target_total_tasks: int,
    min_tasks_per_agent: int,
    max_tasks_per_agent: int,
) -> pd.Series:
    base_labels = source_counts.index.tolist()
    availability = np.array([int(available_counts.get(base, 0)) for base in base_labels], dtype=int)
    if np.any(availability <= 0):
        raise RuntimeError("Ao menos uma base selecionada nao possui tarefas disponiveis em tasks.csv.")

    lower_bounds = np.minimum(np.full(len(base_labels), min_tasks_per_agent, dtype=int), availability)
    upper_bounds = np.minimum(np.full(len(base_labels), max_tasks_per_agent, dtype=int), availability)
    if np.any(lower_bounds > upper_bounds):
        raise RuntimeError("Cotas minimas excedem as cotas maximas/ disponibilidade em alguma base.")

    min_total = int(lower_bounds.sum())
    max_total = int(upper_bounds.sum())
    if target_total_tasks < min_total or target_total_tasks > max_total:
        raise RuntimeError(
            f"target-total-tasks={target_total_tasks} fora da faixa viavel [{min_total}, {max_total}] para os limites informados."
        )

    weights = source_counts.to_numpy(dtype=float)
    weights = weights / weights.sum()
    desired = weights * target_total_tasks
    quotas = lower_bounds.copy()
    remaining = target_total_tasks - int(quotas.sum())

    while remaining > 0:
        eligible = quotas < upper_bounds
        if not np.any(eligible):
            break
        deficits = desired - quotas
        deficits[~eligible] = -np.inf
        chosen = int(np.argmax(deficits))
        quotas[chosen] += 1
        remaining -= 1

    return pd.Series(quotas, index=base_labels, dtype=int)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    source_dir = os.path.normpath(args.source_dir)
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    agents = pd.read_csv(os.path.join(source_dir, "agents.csv"))
    tasks = pd.read_csv(os.path.join(source_dir, "tasks.csv"))
    costs = pd.read_csv(os.path.join(source_dir, "custos.csv"), header=None).values.astype(float)
    resources = pd.read_csv(os.path.join(source_dir, "recursos.csv"), header=None).values.astype(float)

    if "sample_trip_count" not in agents.columns:
        raise RuntimeError("agents.csv nao possui a coluna sample_trip_count.")
    if "original_base" not in tasks.columns:
        raise RuntimeError("tasks.csv nao possui a coluna original_base.")

    source_metadata = load_source_metadata(source_dir)
    capacity_scale = args.capacity_scale if args.capacity_scale > 0 else float(source_metadata.get("capacity_scale", 1.05))

    selected_agents = choose_balanced_bases(agents, args.max_count_ratio, args.min_agents, exact_agents=args.exact_agents)
    selected_bases = selected_agents["dispatching_base_num"].tolist()

    available_counts = tasks["original_base"].value_counts().to_dict()
    min_available = min(int(available_counts.get(base, 0)) for base in selected_bases)
    if min_available <= 0:
        raise RuntimeError("Ao menos uma base selecionada nao possui tarefas disponiveis em tasks.csv.")

    if args.keep_all_tasks:
        selected_tasks = tasks[tasks["original_base"].isin(selected_bases)].copy()
        selected_tasks = selected_tasks.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        tasks_per_agent = 0
        allocation_mode = "keep_all"
        quota_series = selected_tasks["original_base"].value_counts().reindex(selected_bases).astype(int)
    elif args.target_total_tasks > 0:
        if args.min_tasks_per_agent <= 0 or args.max_tasks_per_agent <= 0:
            raise RuntimeError("Use --min-tasks-per-agent e --max-tasks-per-agent quando --target-total-tasks for informado.")
        if args.min_tasks_per_agent > args.max_tasks_per_agent:
            raise RuntimeError("--min-tasks-per-agent nao pode exceder --max-tasks-per-agent.")

        source_counts = selected_agents.set_index("dispatching_base_num")["sample_trip_count"].astype(int)
        quota_series = allocate_controlled_quotas(
            source_counts=source_counts,
            available_counts=available_counts,
            target_total_tasks=args.target_total_tasks,
            min_tasks_per_agent=args.min_tasks_per_agent,
            max_tasks_per_agent=args.max_tasks_per_agent,
        )
        selected_task_parts = []
        for idx, base in enumerate(selected_bases):
            base_tasks = tasks[tasks["original_base"] == base].copy()
            sampled_tasks = base_tasks.sample(n=int(quota_series[base]), random_state=args.seed + idx)
            selected_task_parts.append(sampled_tasks)
        selected_tasks = pd.concat(selected_task_parts, ignore_index=True)
        selected_tasks = selected_tasks.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        tasks_per_agent = 0
        allocation_mode = "controlled_quotas"
    else:
        tasks_per_agent = args.tasks_per_agent if args.tasks_per_agent > 0 else min_available
        if tasks_per_agent > min_available:
            raise RuntimeError(
                f"tasks-per-agent={tasks_per_agent} excede a menor disponibilidade entre as bases selecionadas ({min_available})."
            )

        selected_task_parts = []
        for idx, base in enumerate(selected_bases):
            base_tasks = tasks[tasks["original_base"] == base].copy()
            sampled_tasks = base_tasks.sample(n=tasks_per_agent, random_state=args.seed + idx)
            selected_task_parts.append(sampled_tasks)

        selected_tasks = pd.concat(selected_task_parts, ignore_index=True)
        selected_tasks = selected_tasks.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        allocation_mode = "equal_quotas"
        quota_series = pd.Series(tasks_per_agent, index=selected_bases, dtype=int)

    source_agent_order = agents["dispatching_base_num"].tolist()
    source_task_ids = tasks["task_id"].astype(int).tolist()
    agent_index = {base: idx for idx, base in enumerate(source_agent_order)}
    task_index = {task_id: idx for idx, task_id in enumerate(source_task_ids)}

    row_idx = [agent_index[base] for base in selected_bases]
    col_idx = [task_index[int(task_id)] for task_id in selected_tasks["task_id"].astype(int)]

    balanced_costs = costs[np.ix_(row_idx, col_idx)]
    balanced_resources = resources[np.ix_(row_idx, col_idx)]

    service_factor = selected_agents.set_index("dispatching_base_num")["service_factor"].to_dict()
    capacities = np.zeros(len(selected_bases), dtype=float)
    observed_loads = np.zeros(len(selected_bases), dtype=float)

    for idx, base in enumerate(selected_bases):
        base_mask = selected_tasks["original_base"] == base
        observed_load = float(selected_tasks.loc[base_mask, "duration_min"].sum() * service_factor[base])
        observed_loads[idx] = observed_load
        capacities[idx] = max(observed_load * capacity_scale, observed_load + 10.0)

    min_required_total = float(np.min(balanced_resources, axis=0).sum() * 1.05)
    current_total_capacity = float(capacities.sum())
    if current_total_capacity < min_required_total:
        if observed_loads.sum() > 0:
            weights = observed_loads / observed_loads.sum()
        else:
            weights = np.full(len(selected_bases), 1.0 / len(selected_bases))
        capacities += (min_required_total - current_total_capacity) * weights

    capacities = np.ceil(capacities)

    selected_tasks = selected_tasks.copy()
    selected_tasks["task_id"] = np.arange(len(selected_tasks))

    selected_agents = selected_agents.copy()
    source_counts = selected_agents["sample_trip_count"].astype(int).copy()
    selected_agents["capacity_minutes"] = capacities
    selected_agents["source_sample_trip_count"] = source_counts
    selected_agents["sample_trip_count"] = selected_agents["dispatching_base_num"].map(quota_series.to_dict()).astype(int)

    pd.DataFrame(balanced_costs).to_csv(os.path.join(output_dir, "custos.csv"), header=False, index=False)
    pd.DataFrame(balanced_resources).to_csv(os.path.join(output_dir, "recursos.csv"), header=False, index=False)
    pd.DataFrame(capacities).to_csv(os.path.join(output_dir, "capacidades.csv"), header=False, index=False)
    selected_agents.to_csv(os.path.join(output_dir, "agents.csv"), index=False)
    selected_tasks.to_csv(os.path.join(output_dir, "tasks.csv"), index=False)

    metadata = BalancedSubsetMetadata(
        dataset_name=os.path.basename(output_dir),
        source_type="real_official_tlc_fhv_balanced_subset",
        derived_from_dataset=os.path.basename(source_dir),
        source_metadata_path=os.path.join(source_dir, "metadata.json"),
        num_agents=len(selected_bases),
        num_tasks=len(selected_tasks),
        allocation_mode=allocation_mode,
        capacity_scale=capacity_scale,
        max_count_ratio=args.max_count_ratio,
        min_agents=args.min_agents,
        exact_agents=args.exact_agents,
        target_total_tasks=args.target_total_tasks,
        min_tasks_per_agent=args.min_tasks_per_agent,
        max_tasks_per_agent=args.max_tasks_per_agent,
        tasks_per_agent=tasks_per_agent,
        keep_all_tasks=args.keep_all_tasks,
        random_seed=args.seed,
        selected_bases=selected_bases,
    )
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2, ensure_ascii=False)

    print(f"Instancia balanceada criada em {output_dir}")
    print(f"Base de origem: {source_dir}")
    selected_counts = selected_agents["sample_trip_count"].astype(int)
    source_ratio = float(source_counts.max() / max(1, source_counts.min()))
    final_ratio = float(selected_counts.max() / max(1, selected_counts.min()))
    print(f"Agentes selecionados: {len(selected_bases)} | Tarefas: {len(selected_tasks)}")
    if args.keep_all_tasks:
        print(f"Todas as tarefas das bases selecionadas foram mantidas | Razao max/min final: {final_ratio:.2f}")
    elif args.target_total_tasks > 0:
        print(
            f"Cotas controladas por agente entre {selected_counts.min()} e {selected_counts.max()} | "
            f"Razao max/min final: {final_ratio:.2f}"
        )
    else:
        print(f"Tarefas por agente: {tasks_per_agent} | Razao max/min final: {final_ratio:.2f}")
    print(f"Razao max/min original no subconjunto: {source_ratio:.2f}")
    print("Arquivos gerados: custos.csv, recursos.csv, capacidades.csv, agents.csv, tasks.csv, metadata.json")


if __name__ == "__main__":
    main()
