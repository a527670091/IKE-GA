import argparse
from tqdm import tqdm
import random
import numpy as np
import torch

# 从我们创建的工具文件中导入所需函数
from .utils import (
    load_model, 
    load_data, 
    calculate_fitness, 
    load_corpus_indices,
    dominates,
    non_dominated_sort,
    calculate_crowding_distance
)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Evolutionary In-Context Editing (Evo-ICE)")
    
    # 模型相关参数
    parser.add_argument('--target_model_name', type=str, default='./gpt-j-6B',
                        help='待编辑的目标LLM。可以是在线的HuggingFace模型名，也可以是本地模型路径。')
    parser.add_argument('--evo_model_name', type=str, default='gpt-4',
                        help='用于执行进化操作的LLM (当前版本暂未集成API，此为占位符)。')
    
    # 数据集路径
    parser.add_argument('--data_path', type=str, default='./counterfact.json',
                        help='COUNTERFACT数据集的路径。')

    # 进化算法参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--population_size', type=int, default=50, help='种群大小 (N)。')
    parser.add_argument('--num_generations', type=int, default=20, help='进化代数 (T)。')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='交叉概率。')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='变异概率。')
    parser.add_argument('--k_demos', type=int, default=32, help='每个上下文中的演示数量。')
    parser.add_argument('--batch_size', type=int, default=32, help='评估适应度时的批处理大小。')

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
        line = demo_corpus[demo_id - 2000] 
        
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

def crossover(parent1, parent2, evo_llm):
    """
    步骤 2.2: 交叉 (LLM驱动)
    """
    # TODO: 实现调用EvoLLM进行智能交叉的逻辑
    # print("Performing crossover...")
    child = parent1 # 占位符
    return child

def mutation(individual, evo_llm, demo_corpus):
    """
    步骤 2.3: 变异 (LLM驱动)
    """
    # TODO: 实现调用EvoLLM进行智能变异的逻辑
    # print("Performing mutation...")
    mutated_individual = individual # 占位符
    return mutated_individual

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


def evo_ice_main(args):
    """
    Evo-ICE 主流程
    """
    # 设置随机种子以保证可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 加载模型和数据
    target_llm, tokenizer = load_model(args.target_model_name)
    facts_to_edit, demo_corpus = load_data(args.data_path)
    corpus_indices = load_corpus_indices()

    fact_idx = 0
    fact_to_edit = facts_to_edit[fact_idx] 
    print(f"\nStarting evolution for fact: '{fact_to_edit['requested_rewrite']['prompt'].format(fact_to_edit['requested_rewrite']['subject'])}'")

    population = initialize_population(args.population_size, fact_idx, demo_corpus, corpus_indices, args.k_demos)
    
    # 评估初始种群的适应度
    fitness_scores = []
    for individual in tqdm(population, desc="Evaluating initial population"):
        fitness = calculate_fitness(individual, fact_to_edit, target_llm, tokenizer, args.batch_size)
        fitness_scores.append(fitness)

    for t in range(args.num_generations):
        print(f"\n--- Generation {t+1}/{args.num_generations} ---")
        
        offspring_population = []
        for _ in range(args.population_size // 2):
            parent1, parent2 = selection(population, fitness_scores)
            
            # 交叉和变异的占位符
            if random.random() < args.crossover_rate:
                child1 = crossover(parent1, parent2, None)
            else:
                child1 = parent1

            if random.random() < args.mutation_rate:
                child1 = mutation(child1, None, demo_corpus)
            
            # 生成第二个孩子
            if random.random() < args.crossover_rate:
                child2 = crossover(parent2, parent1, None)
            else:
                child2 = parent2
            
            if random.random() < args.mutation_rate:
                child2 = mutation(child2, None, demo_corpus)

            offspring_population.extend([child1, child2])

        # 评估后代种群
        offspring_fitness = []
        for individual in tqdm(offspring_population, desc=f"Evaluating offspring of generation {t+1}"):
            fitness = calculate_fitness(individual, fact_to_edit, target_llm, tokenizer, args.batch_size)
            offspring_fitness.append(fitness)

        # 更新种群
        combined_population = population + offspring_population
        combined_fitness = fitness_scores + offspring_fitness
        
        population, fitness_scores = update_population(combined_population, combined_fitness, args.population_size)
    
    pareto_front_population, pareto_front_fitness = extract_pareto_front(population, fitness_scores)

    print("\nEvolution finished!")
    print(f"Found {len(pareto_front_population)} solutions on the Pareto front:")
    for i, fitness in enumerate(pareto_front_fitness):
        print(f"  Solution {i+1}: Efficacy={fitness[0]:.4f}, Generalization={fitness[1]:.4f}, Specificity={fitness[2]:.4f}")

if __name__ == '__main__':
    args = parse_args()
    evo_ice_main(args)
