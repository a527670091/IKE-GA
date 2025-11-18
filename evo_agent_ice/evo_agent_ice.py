import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
import json
import os
from datetime import datetime
import time
import pickle
import sys
import multiprocessing as mp
from itertools import repeat

# 设置 multiprocessing 启动方式为 'spawn'（CUDA 兼容）
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# 导入Agent抽象
from .agents import Agent

# 从工具文件中导入所需函数
from .utils import (
    load_model, 
    load_data, 
    calculate_fitness, 
    load_corpus_indices,
    dominates,
    non_dominated_sort,
    calculate_crowding_distance,
    COUNTERFACT_SPLIT_INDEX
)
# 导入LLM操作模块
from .llm_operations import (
    call_evo_llm,
    format_individual_for_prompt,
    get_crossover_prompt,
    get_mutation_prompt,
    parse_llm_response
)

# --- 全局变量，用于工作进程初始化 ---
target_llm_worker = None
tokenizer_worker = None
fact_to_edit_worker = None

def init_worker(model_name, devices, fact_ref, seed):
    """
    工作进程的初始化函数。
    每个工作进程会在这里加载自己的模型副本。
    """
    import multiprocessing
    import os

    global target_llm_worker, tokenizer_worker, fact_to_edit_worker
    
    # 1. 获取工作进程ID并分配GPU
    worker_process = multiprocessing.current_process()
    worker_id = worker_process._identity[0] - 1 if worker_process._identity else 0
    device = devices[worker_id % len(devices)]
    
    print(f"[Evo-Agent] Initializing worker PID {os.getpid()} (Worker ID: {worker_id}) for device {device}...")

    # 2. 在工作进程内部加载模型
    target_llm_worker, tokenizer_worker = load_model(model_name, device)
    fact_to_edit_worker = fact_ref

    # 3. 为每个工作进程设置独立的、确定性的随机种子
    final_seed = seed + worker_id
    np.random.seed(final_seed)
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)
    
    print(f"[Evo-Agent] Worker {worker_id} on device {device} initialized successfully with seed {final_seed}.")


def evaluate_individual_worker(strategy):
    """
    工作进程执行的评估任务。
    注意：这里接收的是Agent.strategy (list[str])，而不是Agent对象本身。
    因为Agent对象可能包含不可pickle的内容。
    """
    return calculate_fitness(
        strategy,
        fact_to_edit_worker,
        target_llm_worker,
        tokenizer_worker
    )

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Evolutionary Agent-based In-Context Editing (Evo-Agent)")
    
    # 模型相关参数
    parser.add_argument('--target_model_name', type=str, default='EleutherAI/gpt-j-6B',
                        help='待编辑的目标LLM。')
    parser.add_argument('--evo_model_name', type=str, default='gemini-2.5-pro',
                        help='用于执行进化操作的LLM (例如: gemini-2.5-pro, gpt-4)。')
    
    # 数据集路径
    parser.add_argument('--data_path', type=str, default='./counterfact.json',
                        help='COUNTERFACT数据集的路径。')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='保存帕累托前沿结果的目录路径。')
    parser.add_argument('--gpu_devices', type=str, default='0',
                        help='用于数据并行的GPU设备ID，用逗号分隔，例如 "0,1,2"。')

    # 进化算法参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--population_size', type=int, default=6, help='种群大小 (N)。')
    parser.add_argument('--num_generations', type=int, default=5, help='进化代数 (T)。')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='交叉概率。')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='变异概率。')
    parser.add_argument('--k_demos', type=int, default=10, help='每个上下文中的演示数量。')
    parser.add_argument('--batch_size', type=int, default=512, help='评估适应度时的批处理大小。')

    args = parser.parse_args()
    return args

def _construct_one_individual(fact_idx: int, demo_corpus: list, corpus_indices: list[list[int]], k: int) -> list[str]:
    """
    构建单个演示上下文（个体的策略）。
    返回 list[str]，将被包装成 Agent.strategy。
    """
    order = [2] * (k // 2) + [1] * (k // 4) + [0] * (k - k // 2 - k // 4)
    random.shuffle(order)
    
    icl_examples = []
    demo_ids = corpus_indices[fact_idx]
    demo_ids = demo_ids[:k]

    for demo_id, o in zip(demo_ids, order):
        line = demo_corpus[demo_id - COUNTERFACT_SPLIT_INDEX] 
        
        rewrite = line['requested_rewrite']
        new_fact_prompt = rewrite['prompt'].format(rewrite['subject'])
        target_new = rewrite['target_new']['str']
        target_true = rewrite['target_true']['str']
        
        if o == 0: # Copy
            icl_examples.append(f'New Fact: {new_fact_prompt} is {target_new}.\nPrompt: {new_fact_prompt} is?\nA: {target_new}\n\n')
        elif o == 1: # Update (Paraphrase)
            prompt = random.choice(line['paraphrase_prompts'])
            icl_examples.append(f'New Fact: {new_fact_prompt} is {target_new}.\nPrompt: {prompt}\nA: {target_new}\n\n')
        elif o == 2: # Retain (Neighborhood)
            prompt = random.choice(line['neighborhood_prompts'])
            icl_examples.append(f'New Fact: {new_fact_prompt} is {target_new}.\nPrompt: {prompt}\nA: {target_true}\n\n')
    
    icl_examples.reverse()
    return icl_examples

def initialize_population(N, fact_idx, demo_corpus, corpus_indices, k):
    """
    步骤 1: 初始化智能体种群
    返回 list[Agent]，每个Agent带有初始strategy，但fitness为None。
    """
    print("[Evo-Agent] Initializing agent population...")
    agents = []
    for i in tqdm(range(N), desc="Generating initial agents"):
        strategy = _construct_one_individual(fact_idx, demo_corpus, corpus_indices, k)
        agent = Agent(strategy=strategy, fitness=None, agent_id=i)
        agents.append(agent)
    return agents

def selection(agents: list[Agent]) -> tuple[Agent, Agent]:
    """
    步骤 2.1: 选择父代 - 基于帕累托支配的二元锦标赛选择法
    直接从Agent对象的fitness属性中获取适应度。
    """
    def _tournament(a1: Agent, a2: Agent) -> Agent:
        if not a1.has_fitness() or not a2.has_fitness():
            raise ValueError("Cannot select agents without fitness evaluation")
        
        if dominates(a1.fitness, a2.fitness):
            return a1
        elif dominates(a2.fitness, a1.fitness):
            return a2
        else:
            return random.choice([a1, a2])

    # 从种群中随机选择4个智能体进行两场锦标赛
    candidates = random.sample(agents, 4)
    parent1 = _tournament(candidates[0], candidates[1])
    parent2 = _tournament(candidates[2], candidates[3])

    return parent1, parent2

def crossover(parent1_strategy: list[str], parent2_strategy: list[str], evo_model_name: str, k_demos: int) -> list[str]:
    """
    步骤 2.2: 交叉 (LLM驱动)
    Phase 1: 仍使用现有的crossover逻辑。
    Phase 2: 将替换为agent_collaborate()。
    """
    print("[Evo-Agent] Performing crossover...")
    try:
        parent1_str = format_individual_for_prompt(parent1_strategy)
        parent2_str = format_individual_for_prompt(parent2_strategy)
        
        prompt = get_crossover_prompt(parent1_str, parent2_str)
        response_text = call_evo_llm(prompt, evo_model_name)
        
        child = parse_llm_response(response_text, k_demos)
        
        if child is None:
            print("[Evo-Agent] Crossover parsing failed, returning parent1 as fallback.")
            return parent1_strategy # 回退机制
        
        print("[Evo-Agent] Crossover successful.")
        return child
    except Exception as e:
        print(f"[Evo-Agent] An error occurred during crossover: {e}")
        return parent1_strategy # 出现任何异常都执行回退

def mutation(individual_strategy: list[str], evo_model_name: str, demo_corpus: list, k_demos: int) -> list[str]:
    """
    步骤 2.3: 变异 (LLM驱动)
    Phase 1: 仍使用现有的mutation逻辑。
    Phase 2: 将替换为agent_self_improve()。
    """
    print("[Evo-Agent] Performing mutation...")
    try:
        individual_str = format_individual_for_prompt(individual_strategy)

        prompt = get_mutation_prompt(individual_str)
        response_text = call_evo_llm(prompt, evo_model_name)
        
        mutated_individual = parse_llm_response(response_text, k_demos)
        
        if mutated_individual is None:
            print("[Evo-Agent] Mutation parsing failed, returning original individual as fallback.")
            return individual_strategy # 回退机制

        print("[Evo-Agent] Mutation successful.")
        return mutated_individual
    except Exception as e:
        print(f"[Evo-Agent] An error occurred during mutation: {e}")
        return individual_strategy # 出现任何异常都执行回退

def update_population(combined_agents: list[Agent], N: int) -> list[Agent]:
    """
    步骤 2.4: 更新种群 (NSGA-II 精英选择策略)
    
    注意：这里combined_agents的每个Agent都已经有fitness了。
    我们需要从combined_agents中提取fitness列表，然后用NSGA-II选出最优的N个Agent。
    """
    # 提取适应度
    combined_fitness = [agent.fitness for agent in combined_agents]
    
    # 使用NSGA-II进行非支配排序
    fronts = non_dominated_sort(combined_agents, combined_fitness)
    
    next_agents = []
    
    for front in fronts:
        if len(next_agents) + len(front) <= N:
            # 整个前沿都可以加入
            for idx in front:
                next_agents.append(combined_agents[idx])
        else:
            # 需要根据拥挤度选择部分个体
            crowding_distances = calculate_crowding_distance(front, combined_fitness)
            sorted_front = sorted(front, key=lambda idx: crowding_distances[idx], reverse=True)
            
            remaining_space = N - len(next_agents)
            for idx in sorted_front[:remaining_space]:
                next_agents.append(combined_agents[idx])
            break
            
    return next_agents

def extract_pareto_front(agents: list[Agent]) -> tuple[list[Agent], list[tuple[float, float, float]]]:
    """
    从最终种群中提取帕累托最优前沿。
    """
    fitness_scores = [agent.fitness for agent in agents]
    fronts = non_dominated_sort(agents, fitness_scores)
    pareto_front_indices = fronts[0] if fronts else []
    
    pareto_agents = [agents[i] for i in pareto_front_indices]
    pareto_fitness = [agents[i].fitness for i in pareto_front_indices]
    
    return pareto_agents, pareto_fitness

def save_final_results(args, agents: list[Agent], start_time, run_output_dir):
    """将最终的帕累托前沿结果保存到JSON文件。"""
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    print(f"\n[Evo-Agent] Total runtime: {elapsed_time_seconds:.2f} seconds")

    output_filepath = os.path.join(run_output_dir, 'results.json')
    
    # 提取帕累托前沿
    pareto_front_agents, pareto_front_fitness = extract_pareto_front(agents)
    
    if not pareto_front_agents:
        print("[Evo-Agent] No solutions on the Pareto front to save.")
        return

    print(f"[Evo-Agent] Found {len(pareto_front_agents)} solutions on the Pareto front.")
    
    results_to_save = {
        'args': vars(args),
        'total_runtime_seconds': elapsed_time_seconds,
        'pareto_front_solutions': []
    }
    for i, (agent, fitness) in enumerate(zip(pareto_front_agents, pareto_front_fitness)):
        print(f"  Solution {i+1}: Efficacy={fitness[0]:.4f}, Generalization={fitness[1]:.4f}, Specificity={fitness[2]:.4f}")
        results_to_save['pareto_front_solutions'].append({
            'id': i,
            'agent_id': agent.agent_id,
            'fitness': {'efficacy': fitness[0], 'generalization': fitness[1], 'specificity': fitness[2]},
            'demonstration_context': agent.strategy
        })
    
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        print(f"\n[Evo-Agent] Results saved to {output_filepath}")
    except Exception as e:
        print(f"\n[Evo-Agent] Error saving results to {output_filepath}: {e}")

def evo_agent_main(args):
    """
    Evo-Agent 主流程 (Phase 1)
    
    Phase 1目标：将种群从 list[list[str]] 改为 list[Agent]，
    但保持算法行为与Evo-ICE完全等价。
    """
    # 初始化历史记录
    history = {
        'generations': [],
        'eval_counts': [],
        'avg_efficacy': [], 'avg_generalization': [], 'avg_specificity': [],
        'best_efficacy': [], 'best_generalization': [], 'best_specificity': []
    }
    total_evaluations = 0

    np.random.seed(args.seed)
    random.seed(args.seed)

    devices = [f'cuda:{i}' for i in args.gpu_devices.split(',') if i.strip()]
    if not torch.cuda.is_available() or not devices:
        devices = ['cpu']
    print(f"[Evo-Agent] Using {len(devices)} GPUs for data parallelism: {devices}")
    
    # 加载数据，但不在主进程中加载模型
    facts_to_edit, demo_corpus = load_data(args.data_path)
    corpus_indices = load_corpus_indices()

    # 简化版流程：我们先针对单个事实进行进化
    fact_idx = 0 
    fact_to_edit = facts_to_edit[fact_idx] 
    print(f"[Evo-Agent] Starting evolution for fact: '{fact_to_edit['requested_rewrite']['prompt'].format(fact_to_edit['requested_rewrite']['subject'])}'")

    # 步骤 1: 初始化智能体种群
    agents = initialize_population(args.population_size, fact_idx, demo_corpus, corpus_indices, args.k_demos)
    
    # --- 多进程池设置 ---
    init_args = (args.target_model_name, devices, fact_to_edit, args.seed)
    
    with mp.Pool(processes=len(devices), initializer=init_worker, initargs=init_args) as pool:
        # 步骤 2.1: 评估初始种群 (Perception)
        print("[Evo-Agent] Evaluating initial population in parallel...")
        strategies = [agent.strategy for agent in agents]
        fitness_scores = list(tqdm(pool.imap(evaluate_individual_worker, strategies), total=len(strategies)))
        
        # 将适应度写回Agent的memory
        for agent, fitness in zip(agents, fitness_scores):
            agent.fitness = fitness

        # --- 性能日志 ---
        scores_np = np.array(fitness_scores)
        avg_scores = np.mean(scores_np, axis=0)
        best_scores = np.max(scores_np, axis=0)
        print(f"\n--- Generation 0 (Initial Population) Performance ---")
        print(f"  - Average: Efficacy={avg_scores[0]:.4f}, Generalization={avg_scores[1]:.4f}, Specificity={avg_scores[2]:.4f}")
        print(f"  - Best:    Efficacy={best_scores[0]:.4f}, Generalization={best_scores[1]:.4f}, Specificity={best_scores[2]:.4f}")
        print("----------------------------------------------------")

        # 记录第0代（初始种群）的数据
        total_evaluations += len(agents)
        history['generations'].append(0)
        history['eval_counts'].append(total_evaluations)
        history['avg_efficacy'].append(avg_scores[0])
        history['avg_generalization'].append(avg_scores[1])
        history['avg_specificity'].append(avg_scores[2])
        history['best_efficacy'].append(best_scores[0])
        history['best_generalization'].append(best_scores[1])
        history['best_specificity'].append(best_scores[2])

        # 步骤 2: 迭代进化
        for t in range(args.num_generations):
            print(f"\n--- Generation {t+1}/{args.num_generations} ---")
            
            # 构造上下文信息（Phase 2）
            context = {
                'relation_id': fact_to_edit['requested_rewrite'].get('relation_id', 'Unknown'),
                'cluster_size': 1,  # 单事实模式
                'generation': t + 1
            }
            
            # Phase 2: 使用 Agent 的主动行为
            offspring_agents = []
            next_agent_id = len(agents)
            
            pbar = tqdm(range(args.population_size // 2), desc="Generating offspring agents")
            for _ in pbar:
                # 选择两个父代智能体
                parent_a, parent_b = selection(agents)
                
                # Phase 2: Agent 协作 (替代交叉)
                if random.random() < args.crossover_rate:
                    child_agent = parent_a.collaborate(
                        parent_b,
                        context,
                        args.evo_model_name,
                        args.k_demos
                    )
                else:
                    child_agent = Agent(strategy=parent_a.strategy.copy(), fitness=None, agent_id=next_agent_id)
                
                # Phase 2: Agent 自我改进 (替代变异)
                if random.random() < args.mutation_rate:
                    # 计算反馈信息
                    current_fitness = [agent.fitness for agent in agents]
                    avg_fitness = tuple(np.mean(np.array(current_fitness), axis=0))
                    best_fitness = tuple(np.max(np.array(current_fitness), axis=0))
                    
                    feedback = {
                        'current_fitness': child_agent.fitness if child_agent.has_fitness() else parent_a.fitness,
                        'avg_fitness': avg_fitness,
                        'best_fitness': best_fitness
                    }
                    
                    child_agent = child_agent.self_improve(
                        feedback,
                        context,
                        args.evo_model_name,
                        args.k_demos
                    )
                
                child_agent.agent_id = next_agent_id
                offspring_agents.append(child_agent)
                next_agent_id += 1
                
                # Child 2 (Phase 2: 同样使用 Agent 方法)
                if random.random() < args.crossover_rate:
                    child2_agent = parent_b.collaborate(
                        parent_a,
                        context,
                        args.evo_model_name,
                        args.k_demos
                    )
                else:
                    child2_agent = Agent(strategy=parent_b.strategy.copy(), fitness=None, agent_id=next_agent_id)
                
                if random.random() < args.mutation_rate:
                    current_fitness = [agent.fitness for agent in agents]
                    avg_fitness = tuple(np.mean(np.array(current_fitness), axis=0))
                    best_fitness = tuple(np.max(np.array(current_fitness), axis=0))
                    
                    feedback = {
                        'current_fitness': child2_agent.fitness if child2_agent.has_fitness() else parent_b.fitness,
                        'avg_fitness': avg_fitness,
                        'best_fitness': best_fitness
                    }
                    
                    child2_agent = child2_agent.self_improve(
                        feedback,
                        context,
                        args.evo_model_name,
                        args.k_demos
                    )
                
                child2_agent.agent_id = next_agent_id
                offspring_agents.append(child2_agent)
                next_agent_id += 1

            # 步骤 2.4: 环境选择 (Selection) - 评估后代
            print(f"[Evo-Agent] Evaluating offspring of generation {t+1} in parallel...")
            offspring_strategies = [agent.strategy for agent in offspring_agents]
            offspring_fitness = list(tqdm(pool.imap(evaluate_individual_worker, offspring_strategies), total=len(offspring_strategies)))
            
            # 将适应度写回offspring agents
            for agent, fitness in zip(offspring_agents, offspring_fitness):
                agent.fitness = fitness

            # 合并父代和子代
            combined_agents = agents + offspring_agents
            
            # 使用NSGA-II选出下一代
            agents = update_population(combined_agents, args.population_size)

            # --- 性能日志 ---
            current_fitness = [agent.fitness for agent in agents]
            scores_np = np.array(current_fitness)
            avg_scores = np.mean(scores_np, axis=0)
            best_scores = np.max(scores_np, axis=0)
            print(f"--- Generation {t+1} Performance ---")
            print(f"  - Average: Efficacy={avg_scores[0]:.4f}, Generalization={avg_scores[1]:.4f}, Specificity={avg_scores[2]:.4f}")
            print(f"  - Best:    Efficacy={best_scores[0]:.4f}, Generalization={best_scores[1]:.4f}, Specificity={best_scores[2]:.4f}")
            print("-----------------------------------")

            # 记录当前代的数据
            total_evaluations += len(offspring_agents)
            history['generations'].append(t + 1)
            history['eval_counts'].append(total_evaluations)
            history['avg_efficacy'].append(avg_scores[0])
            history['avg_generalization'].append(avg_scores[1])
            history['avg_specificity'].append(avg_scores[2])
            history['best_efficacy'].append(best_scores[0])
            history['best_generalization'].append(best_scores[1])
            history['best_specificity'].append(best_scores[2])
    
    # 步骤 3: 返回最终结果
    pareto_front_agents, pareto_front_fitness = extract_pareto_front(agents)

    print("\n[Evo-Agent] Evolution finished!")
    
    # 返回最终的智能体种群和历史记录
    return agents, history


# ============================================================================
# 聚类进化相关函数（Cluster Evolution Support）
# ============================================================================

# 全局变量，用于 cluster worker 初始化
cluster_facts_worker = None

def init_worker_cluster(model_name, devices, cluster_facts, seed):
    """
    Cluster 模式的工作进程初始化函数。
    
    与单 fact 模式的区别：这里接收的是一个 cluster 内的多个 facts。
    """
    import multiprocessing
    import os
    
    global target_llm_worker, tokenizer_worker, cluster_facts_worker
    
    worker_process = multiprocessing.current_process()
    worker_id = worker_process._identity[0] - 1 if worker_process._identity else 0
    device = devices[worker_id % len(devices)]
    
    print(f"[Evo-Agent Cluster Worker] Initializing PID {os.getpid()} on {device}...")
    
    # 加载模型
    target_llm_worker, tokenizer_worker = load_model(model_name, device)
    
    # 保存整个 cluster 的 facts
    cluster_facts_worker = cluster_facts
    
    # 设置随机种子
    final_seed = seed + worker_id
    np.random.seed(final_seed)
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def evaluate_individual_cluster_worker(strategy):
    """
    Cluster 模式的评估 worker。
    
    在整个 cluster 的所有 facts 上评估一个 strategy，返回平均适应度。
    这样进化出的策略对该 cluster 内的所有 facts 都表现良好。
    """
    all_es, all_ps, all_ns = [], [], []
    
    for fact_dict in cluster_facts_worker:
        es, ps, ns = calculate_fitness(
            strategy,
            fact_dict,
            target_llm_worker,
            tokenizer_worker
        )
        all_es.append(es)
        all_ps.append(ps)
        all_ns.append(ns)
    
    # 返回平均值
    return (
        np.mean(all_es),
        np.mean(all_ps),
        np.mean(all_ns)
    )


def evolve_single_cluster(cluster_facts_with_idx, demo_corpus, corpus_indices, args):
    """
    对单个 cluster 内的所有事实进行联合进化。
    
    Args:
        cluster_facts_with_idx: 该 cluster 的所有事实（带 fact_idx）
        demo_corpus: 演示语料库
        corpus_indices: k-NN 索引
        args: 进化参数
    
    Returns:
        result: {
            'pareto_agents': [...],
            'pareto_fitness': [...],
            'history': {...},
            'cluster_info': {...}
        }
    """
    # 提取事实数据
    fact_dicts = [item['fact_data'] for item in cluster_facts_with_idx]
    fact_indices = [item['fact_idx'] for item in cluster_facts_with_idx]
    
    print(f"[Evo-Agent Cluster] Evolving for {len(fact_dicts)} facts in this cluster...")
    
    # 设置设备
    devices = [f'cuda:{i}' for i in args.gpu_devices.split(',') if i.strip()]
    if not torch.cuda.is_available() or not devices:
        devices = ['cpu']
    
    # 初始化种群（使用第一个 fact 的 k-NN 初始化）
    primary_fact_idx = fact_indices[0]
    agents = initialize_population(
        args.population_size,
        primary_fact_idx,
        demo_corpus,
        corpus_indices,
        args.k_demos
    )
    
    # 设置多进程参数
    init_args = (
        args.target_model_name,
        devices,
        fact_dicts,  # 传入整个 cluster 的 facts
        args.seed
    )
    
    # 历史记录
    history = {
        'generations': [],
        'eval_counts': [],
        'avg_efficacy': [],
        'avg_generalization': [],
        'avg_specificity': [],
        'best_efficacy': [],
        'best_generalization': [],
        'best_specificity': []
    }
    
    with mp.Pool(processes=len(devices), initializer=init_worker_cluster, initargs=init_args) as pool:
        # 初始评估
        print("[Evo-Agent Cluster] Evaluating initial population...")
        strategies = [agent.strategy for agent in agents]
        fitness_scores = list(tqdm(
            pool.imap(evaluate_individual_cluster_worker, strategies),
            total=len(strategies),
            desc="Initial eval"
        ))
        
        for agent, fitness in zip(agents, fitness_scores):
            agent.fitness = fitness
        
        total_evaluations = len(agents)
        
        # 记录初始状态
        current_fitness = [agent.fitness for agent in agents]
        scores_np = np.array(current_fitness)
        avg_scores = np.mean(scores_np, axis=0)
        best_scores = np.max(scores_np, axis=0)
        
        history['generations'].append(0)
        history['eval_counts'].append(total_evaluations)
        history['avg_efficacy'].append(avg_scores[0])
        history['avg_generalization'].append(avg_scores[1])
        history['avg_specificity'].append(avg_scores[2])
        history['best_efficacy'].append(best_scores[0])
        history['best_generalization'].append(best_scores[1])
        history['best_specificity'].append(best_scores[2])
        
        # 进化循环
        for t in range(args.num_generations):
            print(f"\n[Evo-Agent Cluster] --- Generation {t+1}/{args.num_generations} ---")
            
            # 构造上下文信息（Phase 2）
            context = {
                'relation_id': cluster_facts_with_idx[0]['fact_data']['requested_rewrite'].get('relation_id', 'Unknown'),
                'cluster_size': len(cluster_facts_with_idx),
                'generation': t + 1
            }
            
            # Phase 2: 使用 Agent 的主动行为
            offspring_agents = []
            for _ in range(args.population_size // 2):
                parent_a, parent_b = selection(agents)
                
                # Phase 2: Agent 协作 (替代交叉)
                if random.random() < args.crossover_rate:
                    child_agent = parent_a.collaborate(
                        parent_b,
                        context,
                        args.evo_model_name,
                        args.k_demos
                    )
                else:
                    child_agent = Agent(strategy=parent_a.strategy.copy())
                
                # Phase 2: Agent 自我改进 (替代变异)
                if random.random() < args.mutation_rate:
                    # 计算反馈信息
                    current_fitness = [agent.fitness for agent in agents]
                    avg_fitness = tuple(np.mean(np.array(current_fitness), axis=0))
                    best_fitness = tuple(np.max(np.array(current_fitness), axis=0))
                    
                    feedback = {
                        'current_fitness': child_agent.fitness if child_agent.has_fitness() else parent_a.fitness,
                        'avg_fitness': avg_fitness,
                        'best_fitness': best_fitness
                    }
                    
                    child_agent = child_agent.self_improve(
                        feedback,
                        context,
                        args.evo_model_name,
                        args.k_demos
                    )
                
                offspring_agents.append(child_agent)
            
            # 评估后代
            offspring_strategies = [agent.strategy for agent in offspring_agents]
            offspring_fitness = list(tqdm(
                pool.imap(evaluate_individual_cluster_worker, offspring_strategies),
                total=len(offspring_strategies),
                desc=f"Gen {t+1} eval"
            ))
            
            for agent, fitness in zip(offspring_agents, offspring_fitness):
                agent.fitness = fitness
            
            # NSGA-II 更新
            combined_agents = agents + offspring_agents
            agents = update_population(combined_agents, args.population_size)
            
            # 性能日志
            current_fitness = [agent.fitness for agent in agents]
            scores_np = np.array(current_fitness)
            avg_scores = np.mean(scores_np, axis=0)
            best_scores = np.max(scores_np, axis=0)
            print(f"[Evo-Agent Cluster] --- Generation {t+1} Performance ---")
            print(f"  - Average: Efficacy={avg_scores[0]:.4f}, Generalization={avg_scores[1]:.4f}, Specificity={avg_scores[2]:.4f}")
            print(f"  - Best:    Efficacy={best_scores[0]:.4f}, Generalization={best_scores[1]:.4f}, Specificity={best_scores[2]:.4f}")
            print("-----------------------------------")
            
            # 记录历史
            total_evaluations += len(offspring_agents)
            history['generations'].append(t + 1)
            history['eval_counts'].append(total_evaluations)
            history['avg_efficacy'].append(avg_scores[0])
            history['avg_generalization'].append(avg_scores[1])
            history['avg_specificity'].append(avg_scores[2])
            history['best_efficacy'].append(best_scores[0])
            history['best_generalization'].append(best_scores[1])
            history['best_specificity'].append(best_scores[2])
    
    # 提取帕累托前沿
    pareto_agents, pareto_fitness = extract_pareto_front(agents)
    
    print(f"[Evo-Agent Cluster] Evolution finished! Pareto front size: {len(pareto_agents)}")
    
    return {
        'agents': agents,
        'pareto_agents': pareto_agents,
        'pareto_fitness': pareto_fitness,
        'history': history,
        'cluster_info': {
            'fact_indices': fact_indices,
            'num_facts': len(fact_dicts)
        }
    }


def evo_agent_main_clustered(args, clusters_to_evolve):
    """
    对多个 clusters 进行分组进化的主控制器。
    
    Args:
        args: 参数对象
        clusters_to_evolve: 要进化的 cluster 列表（已过滤）
    
    Returns:
        all_results: {relation_id: result_dict, ...}
    """
    # 加载数据
    facts_to_edit, demo_corpus = load_data(args.data_path)
    corpus_indices = load_corpus_indices(args.corpus_idx_path)
    
    print(f"\n[Evo-Agent Clustered] Starting evolution for {len(clusters_to_evolve)} clusters...")
    
    all_results = {}
    
    for cluster_idx, cluster in enumerate(clusters_to_evolve):
        relation_id = cluster['relation_id']
        cluster_size = cluster['size']
        cluster_facts = cluster['facts']
        
        print(f"\n{'='*70}")
        print(f"[Evo-Agent Clustered] Cluster {cluster_idx+1}/{len(clusters_to_evolve)}")
        print(f"  Relation ID: {relation_id}")
        print(f"  Size: {cluster_size} facts")
        print(f"{'='*70}")
        
        try:
            result = evolve_single_cluster(
                cluster_facts,
                demo_corpus,
                corpus_indices,
                args
            )
            all_results[relation_id] = result
            
            # 保存单个 cluster 的结果
            cluster_output_dir = os.path.join(args.output_dir, f"cluster_{relation_id}")
            os.makedirs(cluster_output_dir, exist_ok=True)
            
            # 保存帕累托前沿
            pareto_data = {
                'relation_id': relation_id,
                'cluster_size': cluster_size,
                'pareto_front_size': len(result['pareto_agents']),
                'pareto_fitness': result['pareto_fitness'],
                'pareto_strategies': [agent.strategy for agent in result['pareto_agents']]
            }
            with open(os.path.join(cluster_output_dir, 'pareto_front.json'), 'w') as f:
                json.dump(pareto_data, f, indent=2)
            
            print(f"[Evo-Agent Clustered] Cluster {relation_id} results saved to {cluster_output_dir}")
            
        except Exception as e:
            print(f"[Evo-Agent Clustered] Error evolving cluster {relation_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[Evo-Agent Clustered] All clusters evolution finished!")
    print(f"Successfully evolved {len(all_results)}/{len(clusters_to_evolve)} clusters")
    
    return all_results


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    
    # 为直接运行此文件也创建一个唯一的输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir_name = f"evo_agent_{timestamp}_seed{args.seed}_pop{args.population_size}_gens{args.num_generations}_k{args.k_demos}"
    run_output_dir = os.path.join(args.output_dir, run_dir_name)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"--- Evo-Agent Direct Run Output ---")
    print(f"Results for this run will be saved in: {run_output_dir}")
    print("------------------------------------")

    agents, history = evo_agent_main(args)
    save_final_results(args, agents, start_time, run_output_dir)
