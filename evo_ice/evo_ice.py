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

# 从我们创建的工具文件中导入所需函数
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
# 导入新的LLM操作模块
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
    # --- 关键修复：在函数内部导入multiprocessing ---
    # 在 'spawn' 模式下，子进程的环境是全新的，
    # 在函数内部导入可以确保模块在该上下文中一定可用。
    import multiprocessing
    import os

    global target_llm_worker, tokenizer_worker, fact_to_edit_worker
    
    # 1. 获取工作进程ID并分配GPU
    worker_process = multiprocessing.current_process()
    # _identity is a tuple e.g. (1,), we take the first element. Pool processes are 1-indexed.
    worker_id = worker_process._identity[0] - 1 if worker_process._identity else 0
    device = devices[worker_id % len(devices)]
    
    print(f"Initializing worker PID {os.getpid()} (Worker ID: {worker_id}) for device {device}...")

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
    
    print(f"Worker {worker_id} on device {device} initialized successfully with seed {final_seed}.")


def evaluate_individual_worker(individual):
    """
    工作进程执行的评估任务。
    """
    # 使用在初始化时加载到全局变量的模型
    return calculate_fitness(
        individual,
        fact_to_edit_worker,
        target_llm_worker,
        tokenizer_worker
    )

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Evolutionary In-Context Editing (Evo-ICE)")
    
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
    parser.add_argument('--population_size', type=int, default=10, help='种群大小 (N)。')
    parser.add_argument('--num_generations', type=int, default=10, help='进化代数 (T)。')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='交叉概率。')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='变异概率。')
    parser.add_argument('--k_demos', type=int, default=10, help='每个上下文中的演示数量。')
    parser.add_argument('--batch_size', type=int, default=512, help='评估适应度时的批处理大小。')

    args = parser.parse_args()
    return args

def _construct_one_individual(fact_idx: int, demo_corpus: list, corpus_indices: list[list[int]], k: int) -> list[str]:
    """
    借鉴icl.py的逻辑，构建单个演示上下文（个体）。
    """
    order = [2] * (k // 2) + [1] * (k // 4) + [0] * (k - k // 2 - k // 4)
    random.shuffle(order)
    
    icl_examples = []
    demo_ids = corpus_indices[fact_idx]
    demo_ids = demo_ids[:k]

    for demo_id, o in zip(demo_ids, order):
        # demo_id在corpus_idx.txt中是从1开始的，并且demo_corpus是从facts后面开始的
        # 在counterfact.json中，demo_corpus的索引是从COUNTERFACT_SPLIT_INDEX(2000)开始的
        # 因此，(demo_id - COUNTERFACT_SPLIT_INDEX) 可以在0-indexed的demo_corpus中找到正确的演示
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
    步骤 1: 初始化种群
    """
    print("Initializing population...")
    population = []
    for _ in tqdm(range(N), desc="Generating initial individuals"):
        individual = _construct_one_individual(fact_idx, demo_corpus, corpus_indices, k)
        population.append(individual)
    return population

def selection(population, fitness_scores):
    """
    步骤 2.1: 选择父代 - 基于帕累托支配的二元锦标赛选择法
    """
    def _tournament(p1_idx, p2_idx):
        p1_fitness = fitness_scores[p1_idx]
        p2_fitness = fitness_scores[p2_idx]
        
        if dominates(p1_fitness, p2_fitness):
            return p1_idx
        elif dominates(p2_fitness, p1_fitness):
            return p2_idx
        else:
            return random.choice([p1_idx, p2_idx])

    p1_idx, p2_idx = random.sample(range(len(population)), 2)
    winner1_idx = _tournament(p1_idx, p2_idx)
    
    p3_idx, p4_idx = random.sample(range(len(population)), 2)
    winner2_idx = _tournament(p3_idx, p4_idx)

    return population[winner1_idx], population[winner2_idx]

def crossover(parent1: list[str], parent2: list[str], evo_model_name: str, k_demos: int) -> list[str]:
    """
    步骤 2.2: 交叉 (LLM驱动)
    """
    print("Performing crossover...")
    try:
        parent1_str = format_individual_for_prompt(parent1)
        parent2_str = format_individual_for_prompt(parent2)
        
        prompt = get_crossover_prompt(parent1_str, parent2_str)
        response_text = call_evo_llm(prompt, evo_model_name)
        
        child = parse_llm_response(response_text, k_demos)
        
        if child is None:
            print("Crossover parsing failed, returning parent1 as fallback.")
            return parent1 # 回退机制
        
        print("Crossover successful.")
        return child
    except Exception as e:
        print(f"An error occurred during crossover: {e}")
        return parent1 # 出现任何异常都执行回退

def mutation(individual: list[str], evo_model_name: str, demo_corpus: list, k_demos: int) -> list[str]:
    """
    步骤 2.3: 变异 (LLM驱动)
    """
    print("Performing mutation...")
    try:
        individual_str = format_individual_for_prompt(individual)

        prompt = get_mutation_prompt(individual_str)
        response_text = call_evo_llm(prompt, evo_model_name)
        
        mutated_individual = parse_llm_response(response_text, k_demos)
        
        if mutated_individual is None:
            print("Mutation parsing failed, returning original individual as fallback.")
            return individual # 回退机制

        print("Mutation successful.")
        return mutated_individual
    except Exception as e:
        print(f"An error occurred during mutation: {e}")
        return individual # 出现任何异常都执行回退

def update_population(combined_population, combined_fitness, N):
    """
    步骤 2.4: 更新种群 (NSGA-II 精英选择策略)
    """
    fronts = non_dominated_sort(combined_population, combined_fitness)
    
    next_population = []
    next_fitness = []
    
    for front in fronts:
        if len(next_population) + len(front) <= N:
            for idx in front:
                next_population.append(combined_population[idx])
                next_fitness.append(combined_fitness[idx])
        else:
            crowding_distances = calculate_crowding_distance(front, combined_fitness)
            sorted_front = sorted(front, key=lambda idx: crowding_distances[idx], reverse=True)
            
            remaining_space = N - len(next_population)
            for idx in sorted_front[:remaining_space]:
                next_population.append(combined_population[idx])
                next_fitness.append(combined_fitness[idx])
            break
            
    return next_population, next_fitness

def extract_pareto_front(population, fitness_scores):
    """
    从最终种群中提取帕累托最优前沿。
    """
    fronts = non_dominated_sort(population, fitness_scores)
    pareto_front_indices = fronts[0] if fronts else []
    
    pareto_population = [population[i] for i in pareto_front_indices]
    pareto_fitness = [fitness_scores[i] for i in pareto_front_indices]
    
    return pareto_population, pareto_fitness

def save_final_results(args, population, fitness_scores, start_time):
    """将最终的帕累托前沿结果保存到JSON文件。"""
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    print(f"\nTotal runtime: {elapsed_time_seconds:.2f} seconds")

    # 1. 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 构造文件名
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_seed{args.seed}_pop{args.population_size}_results.json"
    output_filepath = os.path.join(args.output_dir, filename)
    
    # 3. 提取帕累托前沿
    pareto_front_population, pareto_front_fitness = extract_pareto_front(population, fitness_scores)
    
    if not pareto_front_population:
        print("No solutions on the Pareto front to save.")
        return

    print(f"Found {len(pareto_front_population)} solutions on the Pareto front.")
    
    results_to_save = {
        'args': vars(args),
        'total_runtime_seconds': elapsed_time_seconds,
        'pareto_front_solutions': []
    }
    for i, (individual, fitness) in enumerate(zip(pareto_front_population, pareto_front_fitness)):
        print(f"  Solution {i+1}: Efficacy={fitness[0]:.4f}, Generalization={fitness[1]:.4f}, Specificity={fitness[2]:.4f}")
        results_to_save['pareto_front_solutions'].append({
            'id': i,
            'fitness': {'efficacy': fitness[0], 'generalization': fitness[1], 'specificity': fitness[2]},
            'demonstration_context': individual
        })
    
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved to {output_filepath}")
    except Exception as e:
        print(f"\nError saving results to {output_filepath}: {e}")

def evo_ice_main(args):
    """
    Evo-ICE 主流程
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
    print(f"Using {len(devices)} GPUs for data parallelism: {devices}")
    
    # 加载数据，但不在主进程中加载模型
    facts_to_edit, demo_corpus = load_data(args.data_path)
    corpus_indices = load_corpus_indices()

    # 简化版流程：我们先针对单个事实进行进化
    fact_idx = 0 
    fact_to_edit = facts_to_edit[fact_idx] 
    print(f"Starting evolution for fact: '{fact_to_edit['requested_rewrite']['prompt'].format(fact_to_edit['requested_rewrite']['subject'])}'")

    # 初始化种群
    population = initialize_population(args.population_size, fact_idx, demo_corpus, corpus_indices, args.k_demos)
    
    # --- 多进程池设置 ---
    # 将模型名称和设备信息作为参数传递给工作进程初始化函数
    init_args = (args.target_model_name, devices, fact_to_edit, args.seed)
    
    with mp.Pool(processes=len(devices), initializer=init_worker, initargs=init_args) as pool:
        print("Evaluating initial population in parallel...")
        fitness_scores = list(tqdm(pool.imap(evaluate_individual_worker, population), total=len(population)))

        # 记录第0代（初始种群）的数据
        total_evaluations += len(population)
        history['generations'].append(0)
        history['eval_counts'].append(total_evaluations)
        scores_np = np.array(fitness_scores)
        avg_scores = np.mean(scores_np, axis=0)
        best_scores = np.max(scores_np, axis=0)
        history['avg_efficacy'].append(avg_scores[0])
        history['avg_generalization'].append(avg_scores[1])
        history['avg_specificity'].append(avg_scores[2])
        history['best_efficacy'].append(best_scores[0])
        history['best_generalization'].append(best_scores[1])
        history['best_specificity'].append(best_scores[2])

        # 迭代进化
        for t in range(args.num_generations):
            print(f"\n--- Generation {t+1}/{args.num_generations} ---")
            
            # 创建后代种群
            offspring_population = []
            pbar = tqdm(range(args.population_size // 2), desc="Generating offspring")
            for _ in pbar:
                parent1, parent2 = selection(population, fitness_scores)
                
                # ... (crossover and mutation logic remains the same)
                if random.random() < args.crossover_rate:
                    child1 = crossover(parent1, parent2, args.evo_model_name, args.k_demos)
                else:
                    child1 = parent1
                if random.random() < args.mutation_rate:
                    child1 = mutation(child1, args.evo_model_name, demo_corpus, args.k_demos)
                if random.random() < args.crossover_rate:
                    child2 = crossover(parent2, parent1, args.evo_model_name, args.k_demos)
                else:
                    child2 = parent2
                if random.random() < args.mutation_rate:
                    child2 = mutation(child2, args.evo_model_name, demo_corpus, args.k_demos)
                offspring_population.extend([child1, child2])

            # 评估后代种群
            print(f"Evaluating offspring of generation {t+1} in parallel...")
            offspring_fitness = list(tqdm(pool.imap(evaluate_individual_worker, offspring_population), total=len(offspring_population)))

            # 更新种群
            combined_population = population + offspring_population
            combined_fitness = fitness_scores + offspring_fitness
            population, fitness_scores = update_population(combined_population, combined_fitness, args.population_size)

            # 记录当前代的数据
            total_evaluations += len(offspring_population)
            history['generations'].append(t + 1)
            history['eval_counts'].append(total_evaluations)
            scores_np = np.array(fitness_scores)
            avg_scores = np.mean(scores_np, axis=0)
            best_scores = np.max(scores_np, axis=0)
            history['avg_efficacy'].append(avg_scores[0])
            history['avg_generalization'].append(avg_scores[1])
            history['avg_specificity'].append(avg_scores[2])
            history['best_efficacy'].append(best_scores[0])
            history['best_generalization'].append(best_scores[1])
            history['best_specificity'].append(best_scores[2])
    
    # 提取帕累托前沿
    pareto_front_population, pareto_front_fitness = extract_pareto_front(population, fitness_scores)

    print("\nEvolution finished!")
    
    # 返回最终的种群、适应度分数和历史记录
    return population, fitness_scores, history


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    # 注意：直接运行此文件时，将不会保存历史记录图表
    population, fitness_scores, history = evo_ice_main(args)
    save_final_results(args, population, fitness_scores, start_time)
