import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Reprodutibilidade bÇ­stica
random.seed(42)
np.random.seed(42)

# 1. Dados
try:
    c = pd.read_csv("data/custos.csv", header=None).values       # custos c(i,j)
    a = pd.read_csv("data/recursos.csv", header=None).values     # recursos a(i,j)
    b = pd.read_csv("data/capacidades.csv", header=None).values.flatten()  # capacidade b(i)
except FileNotFoundError:
    print("Arquivos de dados não encontrados. Usando dados aleatórios para teste.")
    m, n = 5, 50
    c = np.random.rand(m, n) * 10
    a = np.random.rand(m, n) * 5
    b = np.full(m, 100)
else:
    m, n = c.shape  # m agentes, n tarefas

# 2. Funções Objetivo e Viabilidade
def f1(sol):
    """Custo total"""
    return sum(c[sol[j], j] for j in range(n))

def f2(sol):
    """Desequilíbrio de carga"""
    load = np.zeros(m)
    for j in range(n):
        load[sol[j]] += a[sol[j], j]
    return max(load) - min(load)

def is_feasible(sol):
    """Verifica se a solução respeita as capacidades b(i)"""
    load = np.zeros(m)
    for j in range(n):
        load[sol[j]] += a[sol[j], j]
    return np.all(load <= b)

# 3. Heurística Construtiva
def greedy_solution_grasp(alpha=0.3):
    """
    (Item ii-d) Heurística construtiva GRASP.
    Constrói solução inicial com aleatoriedade controlada.
    """
    sol = -np.ones(n, dtype=int)
    load = np.zeros(m)
    tasks = list(range(n))
    random.shuffle(tasks) # Processa as tarefas em ordem aleatória

    for j in tasks:
        # Pega todos os agentes viáveis
        feasible_agents = []
        for i in range(m):
            if load[i] + a[i, j] <= b[i]:
                feasible_agents.append((c[i, j], i)) # (custo, agente)
        
        if not feasible_agents:
            return None 

        # Ordena os agentes viáveis pelo custo
        feasible_agents.sort()
        
        # Constrói a RCL (Restricted Candidate List)
        min_cost = feasible_agents[0][0]
        max_cost_limit = min_cost + alpha * (feasible_agents[-1][0] - min_cost)
        
        RCL = [i for cost, i in feasible_agents if cost <= max_cost_limit]
        
        if not RCL: 
             RCL = [feasible_agents[0][1]]
             
        # Escolhe aleatoriamente um agente da RCL
        chosen_agent = random.choice(RCL)
        
        sol[j] = chosen_agent
        load[chosen_agent] += a[chosen_agent, j]
        
    return sol

# 4. Estruturas de Vizinhança
def neighborhood_shift(sol):
    """
    Vizinhança 1 (Shift): 
    Move uma tarefa 'j' para um agente 'i' aleatório.
    """
    s = sol.copy()
    j = random.randrange(n) # Tarefa aleatória
    current_agent = s[j]
    
    # Escolhe um novo agente aleatório (diferente do atual)
    new_agent = random.randint(0, m - 1)
    while new_agent == current_agent:
        new_agent = random.randint(0, m - 1)
    
    s[j] = new_agent
    return s

def neighborhood_exchange(sol):
    """
    Vizinhança 2 (Exchange): 
    Troca os agentes de duas tarefas 'j1' e 'j2' aleatórias.
    """
    s = sol.copy()
    j1, j2 = random.sample(range(n), 2)
    s[j1], s[j2] = s[j2], s[j1] # Troca os agentes
    return s

def neighborhood_swap(sol):
    """
    Vizinhança 3 (Swap): 
    Troca uma tarefa 'j1' do agente 'i1' por uma tarefa 'j2' do agente 'i2'.
    Movimento mais complexo e poderoso.
    """
    s = sol.copy()
    
    # Encontra dois agentes distintos que tenham tarefas
    agents_with_tasks = list(set(s))
    if len(agents_with_tasks) < 2:
        return s # Não é possível trocar

    i1, i2 = random.sample(agents_with_tasks, 2)

    # Encontra as tarefas atribuídas a cada um
    tasks_i1_indices = np.where(s == i1)[0]
    tasks_i2_indices = np.where(s == i2)[0]

    if len(tasks_i1_indices) == 0 or len(tasks_i2_indices) == 0:
        return s 

    # Sorteia uma tarefa de cada agente
    j1 = random.choice(tasks_i1_indices)
    j2 = random.choice(tasks_i2_indices)

    # Realiza a troca
    s[j1] = i2
    s[j2] = i1
    
    return s

# Lista de vizinhanças para o VNS
neighborhoods = [neighborhood_shift, neighborhood_exchange, neighborhood_swap]

# 5. Estratétgia de Refinamento
# Busca Local Best Improvement REAL

def best_improvement_local_search(sol, obj_func):
    """
    (Item ii-e) Busca local (Best Improvement) usando a vizinhança Shift.
    Testa TODOS os (n * (m-1)) movimentos de shift e escolhe o MELHOR.
    """
    best_sol = sol.copy()
    best_val = obj_func(best_sol)
    
    improved = True
    while improved:
        improved = False
        current_best_move_sol = best_sol # Armazena a melhor solução *desta iteração*
        current_best_move_val = best_val

        # Itera por TODAS as tarefas
        for j in range(n):
            current_agent = best_sol[j]
            
            # Tenta mover a tarefa j para TODOS os outros agentes
            for i in range(m):
                if i == current_agent:
                    continue
                
                candidate_sol = best_sol.copy()
                candidate_sol[j] = i # Realiza o movimento 'shift'
                
                if is_feasible(candidate_sol):
                    candidate_val = obj_func(candidate_sol)
                    
                    # Se este é o melhor movimento encontrado ATÉ AGORA
                    if candidate_val < current_best_move_val:
                        current_best_move_sol = candidate_sol
                        current_best_move_val = candidate_val
        
        # Se, após testar TUDO, encontramos uma melhora
        if current_best_move_val < best_val:
            best_sol, best_val = current_best_move_sol, current_best_move_val
            improved = True # Continua a busca a partir da nova solução
            
    return best_sol, best_val

# 6. Metaheurística VNS
def shake(sol, k):
    """
    Função de Perturbação (Shake) do VNS.
    k=1: usa vizinhança 1 (shift)
    k=2: usa vizinhança 2 (exchange)
    k=3: usa vizinhança 3 (swap)
    k>3: aplica múltiplos movimentos (perturbação mais forte)
    """
    s = sol.copy()
    
    if k == 1:
        s = neighborhood_shift(s)
    elif k == 2:
        s = neighborhood_exchange(s)
    elif k == 3:
        s = neighborhood_swap(s)
    else:
        # Perturbação mais forte: aplica k-2 movimentos 'shift' aleatórios
        for _ in range(k - 2):
            s = neighborhood_shift(s)
    
    # Garante que o shake não gere uma solução inviável
    if not is_feasible(s):
        return sol # Retorna a solução original se o shake falhar
        
    return s

def VNS(obj_func, max_iter=500, k_max=3):
    """
    General Variable Neighborhood Search (GVNS)
    """
    # 1. Solução inicial (Item ii-d)
    sol = None
    while sol is None:
        sol = greedy_solution_grasp(alpha=0.3)
    
    # 2. Refinamento inicial (Item ii-e)
    best_sol, best_val = best_improvement_local_search(sol, obj_func)
    history = [best_val]

    for _ in range(max_iter):
        k = 1
        while k <= k_max:
            # 3. Perturbação (Shake) na vizinhança N_k
            s_shake = shake(best_sol, k)
            
            # 4. Busca Local (Refinamento)
            s_local, val_local = best_improvement_local_search(s_shake, obj_func)
            
            # 5. Critério de Aceitação (Move)
            if val_local < best_val:
                best_sol, best_val = s_local, val_local
                k = 1 # Sucesso! Volta para a primeira vizinhança
            else:
                k = k + 1 # Falha. Tenta uma vizinhança/perturbação maior
        
        history.append(best_val)

    return best_sol, best_val, history

# 7. Execução dos Experimentos e Visualizações
def run_experiments(obj_func, name="f1"):
    results = []
    histories = []
    best_global = None
    best_val_global = float("inf")

    # Cria a pasta de gráficos se não existir
    import os
    base_dir = os.path.join("graphs", "otimizacao_mono_objetivo")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for run in range(5):
        best, val, hist = VNS(obj_func, max_iter=500, k_max=3) # Usando k_max=3
        results.append(val)
        histories.append(hist)
        if val < best_val_global:
            best_global, best_val_global = best, val
        print(f"Execução {run+1} - {name}: {val:.2f}")

    print(f"{name} -> min: {np.min(results):.2f}, mean: {np.mean(results):.2f}, "
          f"std: {np.std(results):.2f}, max: {np.max(results):.2f}")

    # Curvas de convergência
    plt.figure()
    for hist in histories:
        plt.plot(hist, alpha=0.7)
    plt.title(f"Convergência - {name}")
    plt.xlabel("Iteração")
    plt.ylabel(name)
    plt.grid()
    plt.savefig(os.path.join(base_dir, f"{name}_convergencia.png"), dpi=150)
    plt.close()

    # Melhor solução encontrada (carga por agente)
    loads = np.zeros(m)
    for j in range(n):
        loads[best_global[j]] += a[best_global[j], j]

    plt.figure()
    plt.bar(range(1, m+1), loads, tick_label=[f"Agente {i+1}" for i in range(m)])
    plt.title(f"Melhor solução - {name}\nValor: {best_val_global:.2f}")
    plt.xlabel("Agentes")
    plt.ylabel("Carga total")
    plt.grid(axis="y")
    plt.savefig(os.path.join(base_dir, f"{name}_melhor.png"), dpi=150)
    plt.close()

    return results, histories, best_global, best_val_global

# Main
if __name__ == "__main__":
    print("=== Otimização f1 (Custo) ===")
    run_experiments(f1, "f1")

    print("\n=== Otimização f2 (Equilíbrio) ===")
    run_experiments(f2, "f2")
