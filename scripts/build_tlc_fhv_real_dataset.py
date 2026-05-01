import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List
from urllib.parse import urlencode

import numpy as np
import pandas as pd


TRIPS_BASE_URL = "https://data.cityofnewyork.us/resource/vgi6-tcdb.csv"
BASES_URL = "https://data.cityofnewyork.us/api/views/eccv-9dzr/rows.csv?accessType=DOWNLOAD"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


@dataclass
class DatasetMetadata:
    dataset_name: str
    source_type: str
    trip_source: str
    bases_source: str
    zones_source: str
    pickup_start: str
    pickup_end: str
    rows_downloaded: int
    rows_after_cleaning: int
    num_agents: int
    num_tasks: int
    capacity_scale: float
    affinity_penalty_scale: float
    resource_penalty_scale: float
    skip_top_bases: int
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constroi uma instancia real de GAP a partir de viagens TLC/FHV.")
    parser.add_argument("--output-dir", default="data/tlc_fhv_real_2022_01_03", help="Diretorio de saida da instancia.")
    parser.add_argument("--pickup-start", default="2022-01-03T08:00:00", help="Inicio da janela de coleta.")
    parser.add_argument("--pickup-end", default="2022-01-03T12:00:00", help="Fim da janela de coleta.")
    parser.add_argument("--row-limit", type=int, default=12000, help="Numero maximo de linhas baixadas da API.")
    parser.add_argument("--num-agents", type=int, default=5, help="Numero de bases TLC a usar como agentes.")
    parser.add_argument("--num-tasks", type=int, default=60, help="Numero de viagens a usar como tarefas.")
    parser.add_argument(
        "--skip-top-bases",
        type=int,
        default=0,
        help="Numero de bases mais frequentes a ignorar antes de selecionar os agentes.",
    )
    parser.add_argument("--capacity-scale", type=float, default=1.15, help="Folga multiplicativa das capacidades.")
    parser.add_argument("--affinity-penalty-scale", type=float, default=6.0, help="Peso da penalidade de afinidade no custo.")
    parser.add_argument("--resource-penalty-scale", type=float, default=2.0, help="Peso da penalidade de afinidade no recurso.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria.")
    return parser.parse_args()


def build_query_url(start: str, end: str, row_limit: int) -> str:
    params = {
        "$select": "dispatching_base_num,pickup_datetime,dropoff_datetime,pulocationid,dolocationid,affiliated_base_number",
        "$where": f"pickup_datetime between '{start}' and '{end}'",
        "$order": "pickup_datetime",
        "$limit": str(row_limit),
    }
    return f"{TRIPS_BASE_URL}?{urlencode(params)}"


def stratified_sample(df: pd.DataFrame, group_col: str, total_n: int, seed: int) -> pd.DataFrame:
    counts = df[group_col].value_counts().sort_index()
    proportions = counts / counts.sum()
    allocations = {}
    for key, count in counts.items():
        allocations[key] = max(1, int(round(proportions[key] * total_n)))
        allocations[key] = min(allocations[key], int(count))

    allocated = sum(allocations.values())
    if allocated > total_n:
        for key in sorted(allocations, key=allocations.get, reverse=True):
            while allocations[key] > 1 and allocated > total_n:
                allocations[key] -= 1
                allocated -= 1
    elif allocated < total_n:
        remaining = total_n - allocated
        spare = counts.to_dict()
        while remaining > 0:
            progressed = False
            for key in counts.index:
                if allocations[key] < spare[key]:
                    allocations[key] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break

    parts = []
    for idx, (key, take_n) in enumerate(allocations.items()):
        sample = df[df[group_col] == key].sample(n=take_n, random_state=seed + idx)
        parts.append(sample)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    trips_url = build_query_url(args.pickup_start, args.pickup_end, args.row_limit)
    trips = pd.read_csv(trips_url)
    rows_downloaded = len(trips)

    trips = trips.dropna(subset=["dispatching_base_num", "pickup_datetime", "dropoff_datetime", "pulocationid", "dolocationid"]).copy()
    trips["pickup_datetime"] = pd.to_datetime(trips["pickup_datetime"], utc=False)
    trips["dropoff_datetime"] = pd.to_datetime(trips["dropoff_datetime"], utc=False)
    trips["pulocationid"] = trips["pulocationid"].astype(float).astype(int)
    trips["dolocationid"] = trips["dolocationid"].astype(float).astype(int)
    trips["duration_min"] = (trips["dropoff_datetime"] - trips["pickup_datetime"]).dt.total_seconds() / 60.0
    trips = trips[(trips["duration_min"] >= 3.0) & (trips["duration_min"] <= 90.0)].copy()

    base_counts = trips["dispatching_base_num"].value_counts()
    available_bases = base_counts.index.tolist()
    top_bases = available_bases[args.skip_top_bases : args.skip_top_bases + args.num_agents]
    if len(top_bases) < args.num_agents:
        raise RuntimeError(
            f"Nao ha bases suficientes apos ignorar as {args.skip_top_bases} mais frequentes. "
            f"Disponiveis: {len(available_bases)}."
        )
    trips = trips[trips["dispatching_base_num"].isin(top_bases)].copy()
    if len(trips) < args.num_tasks:
        raise RuntimeError(f"Nao ha viagens suficientes apos a filtragem: {len(trips)} disponiveis.")

    sampled = stratified_sample(trips, "dispatching_base_num", args.num_tasks, args.seed)
    selected_bases = sampled["dispatching_base_num"].value_counts().index.tolist()
    sampled = sampled[sampled["dispatching_base_num"].isin(selected_bases)].copy()

    bases = pd.read_csv(BASES_URL)
    bases = bases.rename(columns={"License Number": "dispatching_base_num", "Entity Name": "entity_name", "Type of Base": "base_type"})
    bases = bases[["dispatching_base_num", "entity_name", "base_type", "Street", "City", "State", "Postcode"]]
    base_info = bases[bases["dispatching_base_num"].isin(selected_bases)].drop_duplicates("dispatching_base_num")

    zones = pd.read_csv(ZONES_URL)
    zones = zones.rename(columns={"LocationID": "location_id", "Borough": "borough", "Zone": "zone_name"})

    sampled["original_base"] = sampled["dispatching_base_num"]
    agent_labels = sorted(selected_bases)
    task_records = sampled.reset_index(drop=True).copy()
    num_agents = len(agent_labels)
    num_tasks = len(task_records)

    zone_ids = sorted(set(task_records["pulocationid"]).union(set(task_records["dolocationid"])))
    num_zones = max(1, len(zone_ids))

    pickup_freq: Dict[str, Dict[int, int]] = {}
    dropoff_freq: Dict[str, Dict[int, int]] = {}
    total_freq: Dict[str, int] = {}
    avg_duration: Dict[str, float] = {}

    for base in agent_labels:
        subset = trips[trips["dispatching_base_num"] == base]
        pickup_freq[base] = subset["pulocationid"].value_counts().to_dict()
        dropoff_freq[base] = subset["dolocationid"].value_counts().to_dict()
        total_freq[base] = len(subset)
        avg_duration[base] = float(subset["duration_min"].mean())

    global_avg_duration = float(trips["duration_min"].mean())
    service_factor: Dict[str, float] = {}
    for base in agent_labels:
        factor = avg_duration[base] / max(global_avg_duration, 1e-6)
        service_factor[base] = float(np.clip(factor, 0.85, 1.15))

    costs = np.zeros((num_agents, num_tasks), dtype=float)
    resources = np.zeros((num_agents, num_tasks), dtype=float)

    for i, base in enumerate(agent_labels):
        total_base = total_freq[base]
        for j, row in task_records.iterrows():
            pickup_prob = (pickup_freq[base].get(int(row["pulocationid"]), 0) + 1.0) / (total_base + num_zones)
            dropoff_prob = (dropoff_freq[base].get(int(row["dolocationid"]), 0) + 1.0) / (total_base + num_zones)
            affinity_penalty = -math.log(pickup_prob) - 0.5 * math.log(dropoff_prob)
            duration = float(row["duration_min"])
            costs[i, j] = duration + args.affinity_penalty_scale * affinity_penalty + 0.05 * avg_duration[base]
            resources[i, j] = duration * service_factor[base] + args.resource_penalty_scale * 0.10 * affinity_penalty

    capacities = np.zeros(num_agents, dtype=float)
    observed_loads = []
    for i, base in enumerate(agent_labels):
        original_mask = task_records["original_base"] == base
        observed_load = float(task_records.loc[original_mask, "duration_min"].sum() * service_factor[base])
        observed_loads.append(observed_load)
        capacities[i] = max(observed_load * args.capacity_scale, observed_load + 10.0)

    observed_loads = np.array(observed_loads, dtype=float)
    min_required_total = float(np.min(resources, axis=0).sum() * 1.05)
    current_total_capacity = float(capacities.sum())
    if current_total_capacity < min_required_total:
        if observed_loads.sum() > 0:
            weights = observed_loads / observed_loads.sum()
        else:
            weights = np.full(num_agents, 1.0 / num_agents)
        capacities += (min_required_total - current_total_capacity) * weights

    capacities = np.ceil(capacities)

    pd.DataFrame(costs).to_csv(os.path.join(args.output_dir, "custos.csv"), header=False, index=False)
    pd.DataFrame(resources).to_csv(os.path.join(args.output_dir, "recursos.csv"), header=False, index=False)
    pd.DataFrame(capacities).to_csv(os.path.join(args.output_dir, "capacidades.csv"), header=False, index=False)

    zone_lookup = zones.rename(columns={"location_id": "pulocationid", "borough": "pu_borough", "zone_name": "pu_zone_name"})
    task_export = task_records.merge(zone_lookup, how="left", on="pulocationid")
    zone_lookup_do = zones.rename(columns={"location_id": "dolocationid", "borough": "do_borough", "zone_name": "do_zone_name"})
    task_export = task_export.merge(zone_lookup_do, how="left", on="dolocationid")
    task_export["task_id"] = np.arange(num_tasks)
    task_export.to_csv(os.path.join(args.output_dir, "tasks.csv"), index=False)

    base_export = pd.DataFrame({"dispatching_base_num": agent_labels, "capacity_minutes": capacities})
    base_export = base_export.merge(base_info, how="left", on="dispatching_base_num")
    base_export["mean_duration_min"] = base_export["dispatching_base_num"].map(avg_duration)
    base_export["service_factor"] = base_export["dispatching_base_num"].map(service_factor)
    base_export["sample_trip_count"] = base_export["dispatching_base_num"].map(task_records["original_base"].value_counts().to_dict())
    base_export.to_csv(os.path.join(args.output_dir, "agents.csv"), index=False)

    metadata = DatasetMetadata(
        dataset_name=os.path.basename(os.path.normpath(args.output_dir)),
        source_type="real_official_tlc_fhv",
        trip_source=trips_url,
        bases_source=BASES_URL,
        zones_source=ZONES_URL,
        pickup_start=args.pickup_start,
        pickup_end=args.pickup_end,
        rows_downloaded=rows_downloaded,
        rows_after_cleaning=len(trips),
        num_agents=num_agents,
        num_tasks=num_tasks,
        capacity_scale=args.capacity_scale,
        affinity_penalty_scale=args.affinity_penalty_scale,
        resource_penalty_scale=args.resource_penalty_scale,
        skip_top_bases=args.skip_top_bases,
        random_seed=args.seed,
    )
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2, ensure_ascii=False)

    print(f"Instancia real criada em {args.output_dir}")
    print(f"Agentes: {num_agents} | Tarefas: {num_tasks}")
    print(f"Linhas baixadas: {rows_downloaded} | Linhas limpas: {len(trips)}")
    print("Arquivos gerados: custos.csv, recursos.csv, capacidades.csv, agents.csv, tasks.csv, metadata.json")


if __name__ == "__main__":
    main()
