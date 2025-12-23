import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import time

# Reprodutibilidade bÇ­stica
random.seed(42)
np.random.seed(42)

# 1. Dados
try:
    c = pd.read_csv("data/custos.csv", header=None).values
    a = pd.read_csv("data/recursos.csv", header=None).values
    b = pd.read_csv("data/capacidades.csv", header=None).values.flatten()
except FileNotFoundError:
    print("Arquivos de dados não encontrados. Usando dados aleatórios para teste.")
    m, n = 5, 50
    c = np.random.rand(m, n) * 10
    a = np.random.rand(m, n) * 5
    b = np.full(m, 100)
else:
    m, n = c.shape

# 2. Funções Objetivo e Viabilidade
def f1(sol):
    """Custo total"""
    return sum(c[sol[j], j] for j in range(n))

def f2(sol):
    """Desequilíbrio de carga"""
    load = np.zeros(m)
    for j in range(n):
        load[sol[j]] += a[sol[j], j]
    
    if np.any(load > b): # Checagem extra de viabilidade
        return float('inf') # Solução inviável tem custo infinito
        
    return max(load) - min(load)

def is_feasible(sol):
    """Verifica se a solução respeita as capacidades b(i)"""
    load = np.zeros(m)
    for j in range(n):
        load[sol[j]] += a[sol[j], j]
    return np.all(load <= b)

# 3. Heurística Construtiva
def greedy_solution_grasp(alpha=0.3):
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
        
        RCL = [i for cost, i in feasible_agents if cost <= max_cost_limit]
        
        if not RCL: 
             RCL = [feasible_agents[0][1]]
             
        chosen_agent = random.choice(RCL)
        sol[j] = chosen_agent
        load[chosen_agent] += a[chosen_agent, j]
        
    return sol

# 4. Estruturas de Vizinhança
def neighborhood_shift(sol):
    s = sol.copy()
    j = random.randrange(n)
    current_agent = s[j]
    new_agent = random.randint(0, m - 1)
    while new_agent == current_agent:
        new_agent = random.randint(0, m - 1)
    s[j] = new_agent
    return s

def neighborhood_exchange(sol):
    s = sol.copy()
    j1, j2 = random.sample(range(n), 2)
    s[j1], s[j2] = s[j2], s[j1]
    return s

def neighborhood_swap(sol):
    s = sol.copy()
    agents_with_tasks = list(set(s))
    if len(agents_with_tasks) < 2: return s
    i1, i2 = random.sample(agents_with_tasks, 2)
    tasks_i1_indices = np.where(s == i1)[0]
    tasks_i2_indices = np.where(s == i2)[0]
    if len(tasks_i1_indices) == 0 or len(tasks_i2_indices) == 0: return s 
    j1 = random.choice(tasks_i1_indices)
    j2 = random.choice(tasks_i2_indices)
    s[j1] = i2
    s[j2] = i1
    return s

# 5. Estratétgia de Refinamento
def best_improvement_local_search(sol, obj_func):
    best_sol = sol.copy()
    best_val = obj_func(best_sol)
    
    improved = True
    while improved:
        improved = False
        current_best_move_sol = best_sol
        current_best_move_val = best_val

        for j in range(n):
            current_agent = best_sol[j]
            for i in range(m):
                if i == current_agent: continue
                
                candidate_sol = best_sol.copy()
                candidate_sol[j] = i
                
                if is_feasible(candidate_sol): # A viabilidade básica é sempre checada
                    candidate_val = obj_func(candidate_sol)
                    if candidate_val < current_best_move_val:
                        current_best_move_sol = candidate_sol
                        current_best_move_val = candidate_val
        
        if current_best_move_val < best_val:
            best_sol, best_val = current_best_move_sol, current_best_move_val
            improved = True
            
    return best_sol, best_val

# 6. Metaheurística VNS
def shake(sol, k):
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

def VNS(obj_func, max_iter=500, k_max=3):
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
                k = k + 1
        
        history.append(best_val)

    return best_sol, best_val, history

# 7. Funções De Escalarização

def obj_func_ponderada(sol, w1, f1_min, f1_max, f2_min, f2_max):
    """
    (Entrega 2) Função objetivo para Soma Ponderada (Pw) com normalização.
    """
    val_f1 = f1(sol)
    val_f2 = f2(sol)
    
    # Se a solução for inviável para f2 (retorna inf),
    # retorne inf para o VNS descartá-la.
    if val_f2 == float('inf'):
        return float('inf')

    # Normalização [0, 1]
    # Adicionamos 1e-6 para evitar divisão por zero se min == max
    norm_f1 = (val_f1 - f1_min) / (f1_max - f1_min + 1e-6)
    norm_f2 = (val_f2 - f2_min) / (f2_max - f2_min + 1e-6)
    
    w2 = 1.0 - w1
    return (w1 * norm_f1) + (w2 * norm_f2)

def obj_func_epsilon(sol, epsilon_val):
    """
    (Entrega 2) Função objetivo para Epsilon-Restrito (PE) com penalidade.
    Otimiza f1 e penaliza a violação de f2 <= epsilon_val.
    """
    val_f1 = f1(sol)
    val_f2 = f2(sol)
    
    # Se a solução for inviável (capacidade geral)
    if val_f1 == float('inf') or val_f2 == float('inf'):
        return float('inf')
    
    # Se a restrição epsilon for violada
    if val_f2 > epsilon_val:
        # Penalidade alta: custo + 1000 * (quanto violou)
        penalidade = 1000 * (val_f2 - epsilon_val)
        return val_f1 + penalidade
    else:
        # Se for viável, retorna apenas o custo
        return val_f1

# 8. Funções de Execução

def get_nadir_points():
    """
    (Entrega 2) Executa a otimização mono-objetivo (Entrega 1)
    para encontrar os pontos ideais e nadir para normalização.
    """
    print("Executando análise mono-objetivo para encontrar pontos ideais e nadir...")
    
    # 1. Otimiza f1 (Custo)
    best_sol_f1, f1_min, _ = VNS(f1, max_iter=500, k_max=3)
    # Calcula f2 para a melhor solução de f1 (ponto f2_max)
    f2_max = f2(best_sol_f1)
    
    # 2. Otimiza f2 (Equilíbrio)
    best_sol_f2, f2_min, _ = VNS(f2, max_iter=500, k_max=3)
    # Calcula f1 para a melhor solução de f2 (ponto f1_max)
    f1_max = f1(best_sol_f2)
    
    print(f"  Ponto ideal f1 (custo): {f1_min:.2f} (com f2={f2_max:.2f})")
    print(f"  Ponto ideal f2 (equil.): {f2_min:.2f} (com f1={f1_max:.2f})")
    
    return f1_min, f1_max, f2_min, f2_max

def run_soma_ponderada(f1_min, f1_max, f2_min, f2_max, num_steps=20):
    """
    (Entrega 2) Executa o VNS para a abordagem Soma Ponderada.
    """
    pareto_front_vals = []
    
    # Varia o peso w1 (peso do custo) de 0.0 a 1.0
    for w1 in np.linspace(0.0, 1.0, num_steps):
        # Cria a função objetivo para este peso específico
        obj_func = lambda sol: obj_func_ponderada(sol, w1, f1_min, f1_max, f2_min, f2_max)
        
        # Executa o VNS
        best_sol, _, _ = VNS(obj_func, max_iter=250, k_max=3) # Menos iterações
        
        # Salva os valores REAIS (não normalizados) da solução
        pareto_front_vals.append((f1(best_sol), f2(best_sol)))
        
    return pareto_front_vals

def run_epsilon_restrito(f1_min, f1_max, f2_min, f2_max, num_steps=20):
    """
    (Entrega 2) Executa o VNS para a abordagem Epsilon-Restrito.
    """
    pareto_front_vals = []
    
    # Varia o epsilon de f2_max (pior equilíbrio) até f2_min (equilíbrio perfeito)
    for epsilon_val in np.linspace(f2_max, f2_min, num_steps):
        # Cria a função objetivo para este epsilon específico
        obj_func = lambda sol: obj_func_epsilon(sol, epsilon_val)
        
        # Executa o VNS
        best_sol, best_val_penalizado, _ = VNS(obj_func, max_iter=250, k_max=3)
        
        # Se o VNS encontrou uma solução viável (sem penalidade infinita)
        if best_val_penalizado < float('inf'):
            # Salva os valores REAIS
            pareto_front_vals.append((f1(best_sol), f2(best_sol)))
            
    return pareto_front_vals

def filter_non_dominated(points):
    """
    Filtra uma lista de pontos, mantendo apenas os não-dominados.
    """
    points = sorted(points) # Ordena por f1
    filtered_pareto = []
    
    for point in points:
        is_dominated = False
        # Compara com os pontos já adicionados
        for pareto_point in filtered_pareto:
            # Se um ponto 'pareto_point' é melhor ou igual nos dois objetivos...
            if pareto_point[0] <= point[0] and pareto_point[1] <= point[1]:
                # ... e estritamente melhor em pelo menos um ...
                if pareto_point[0] < point[0] or pareto_point[1] < point[1]:
                    is_dominated = True
                    break
        
        if not is_dominated:
            # Remove pontos da fronteira que são dominados pelo 'point' atual
            filtered_pareto = [p for p in filtered_pareto if not (point[0] <= p[0] and point[1] <= p[1])]
            filtered_pareto.append(point)
            
    return filtered_pareto

# 9. Main

if __name__ == "__main__":

    base_dir = os.path.join("graphs", "otimizacao_multiobjetivo")
    os.makedirs(base_dir, exist_ok=True)

    NUM_EXECUCOES = 5
    NUM_PONTOS_FRONTEIRA = 20  # 20 passos para w e epsilon

    all_fronts_pw = []
    all_fronts_pe = []

    print("=== INICIANDO ENT2 (multiobjetivo) ===")
    start_time = time.time()

    for run in range(NUM_EXECUCOES):
        print(f"\\n--- Execucao {run+1} / {NUM_EXECUCOES} ---")

        f1_min, f1_max, f2_min, f2_max = get_nadir_points()

        print("  Rodando Soma Ponderada (Pw)...")
        front_pw = run_soma_ponderada(f1_min, f1_max, f2_min, f2_max, NUM_PONTOS_FRONTEIRA)
        all_fronts_pw.append(front_pw)

        print("  Rodando Epsilon-Restrito (PE)...")
        front_pe = run_epsilon_restrito(f1_min, f1_max, f2_min, f2_max, NUM_PONTOS_FRONTEIRA)
        all_fronts_pe.append(front_pe)

    print(f"\\nTempo total de execucao: {(time.time() - start_time):.2f} segundos")

    print("Gerando graficos...")

    plt.figure(figsize=(10, 7))
    all_points_pw = []
    for i, front in enumerate(all_fronts_pw):
        x_vals = [p[0] for p in front]
        y_vals = [p[1] for p in front]
        all_points_pw.extend(front)
        plt.scatter(x_vals, y_vals, alpha=0.6, label=f"Execucao {i+1}")
        plt.plot(sorted(x_vals), sorted(y_vals, reverse=True), alpha=0.3)

    final_pareto_pw = filter_non_dominated(all_points_pw)
    if len(final_pareto_pw) > 20:
        indices = np.linspace(0, len(final_pareto_pw) - 1, 20, dtype=int)
        final_pareto_pw = [final_pareto_pw[i] for i in indices]

    plt.scatter([p[0] for p in final_pareto_pw], [p[1] for p in final_pareto_pw], color='red', marker='x', s=100, label="Fronteira Final (20 pontos)")

    plt.title(f"Soma Ponderada (Pw) - {NUM_EXECUCOES} execucoes")
    plt.xlabel("f1 (Custo Total)")
    plt.ylabel("f2 (Desequilibrio de Carga)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, "fronteira_pw.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 7))
    all_points_pe = []
    for i, front in enumerate(all_fronts_pe):
        x_vals = [p[0] for p in front]
        y_vals = [p[1] for p in front]
        all_points_pe.extend(front)
        plt.scatter(x_vals, y_vals, alpha=0.6, label=f"Execucao {i+1}")
        plt.plot(sorted(x_vals), sorted(y_vals, reverse=True), alpha=0.3)

    final_pareto_pe = filter_non_dominated(all_points_pe)
    if len(final_pareto_pe) > 20:
        indices = np.linspace(0, len(final_pareto_pe) - 1, 20, dtype=int)
        final_pareto_pe = [final_pareto_pe[i] for i in indices]

    plt.scatter([p[0] for p in final_pareto_pe], [p[1] for p in final_pareto_pe], color='red', marker='x', s=100, label="Fronteira Final (20 pontos)")

    plt.title(f"Metodo $\epsilon$-Restrito (PE) - {NUM_EXECUCOES} execucoes")
    plt.xlabel("f1 (Custo Total)")
    plt.ylabel("f2 (Desequilibrio de Carga)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, "fronteira_pe.png"), dpi=150)
    plt.close()

    print(f"Graficos 'fronteira_pw.png' e 'fronteira_pe.png' salvos em {base_dir}.")
