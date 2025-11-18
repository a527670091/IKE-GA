"""
Evo-Agent 聚类进化入口脚本

基于 relation_id 对事实进行聚类，然后对每个 cluster 独立进化。
这样可以在保持效率的同时，为不同类型的关系找到最优的编辑策略。
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evo_agent_ice.evo_agent_ice import evo_agent_main_clustered
from evo_agent_ice.clustering import (
    cluster_facts_by_relation,
    print_cluster_statistics,
    filter_clusters,
    save_cluster_info
)
from evo_agent_ice.utils import load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Evo-Agent Clustered Evolution")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./counterfact.json',
                        help='Path to the CounterFact dataset')
    parser.add_argument('--corpus_idx_path', type=str, default='./corpus_idx.txt',
                        help='Path to the corpus indices file')
    
    # 模型参数
    parser.add_argument('--target_model_name', type=str, default='EleutherAI/gpt-j-6B',
                        help='Target model to edit (e.g., EleutherAI/gpt-j-6B)')
    parser.add_argument('--evo_model_name', type=str, default='gemini-2.0-flash-exp',
                        help='LLM for evolution operations (e.g., gemini-2.0-flash-exp)')
    
    # 进化算法参数
    parser.add_argument('--population_size', type=int, default=6,
                        help='Population size for evolution')
    parser.add_argument('--num_generations', type=int, default=5,
                        help='Number of generations')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                        help='Crossover rate')
    parser.add_argument('--mutation_rate', type=float, default=0.2,
                        help='Mutation rate')
    parser.add_argument('--k_demos', type=int, default=10,
                        help='Number of demonstrations in each strategy')
    
    # 聚类参数
    parser.add_argument('--min_cluster_size', type=int, default=5,
                        help='Minimum cluster size to evolve (skip smaller clusters)')
    parser.add_argument('--max_cluster_size', type=int, default=None,
                        help='Maximum cluster size (for limiting computation)')
    parser.add_argument('--top_k_clusters', type=int, default=None,
                        help='Only evolve top K largest clusters (None = all)')
    
    # 系统参数
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu_devices', type=str, default='0',
                        help='GPU devices to use (comma-separated, e.g., "0,1,2")')
    parser.add_argument('--output_dir', type=str, default='./results_clustered',
                        help='Output directory for results')
    
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    
    # 创建唯一的输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir_name = f"clustered_{timestamp}_seed{args.seed}_pop{args.population_size}_gens{args.num_generations}"
    args.output_dir = os.path.join(args.output_dir, run_dir_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Evo-Agent Clustered Evolution")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Target model: {args.target_model_name}")
    print(f"Evolution model: {args.evo_model_name}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.num_generations}")
    print(f"Min cluster size: {args.min_cluster_size}")
    if args.top_k_clusters:
        print(f"Top K clusters: {args.top_k_clusters}")
    print("="*70)
    
    # Step 1: 加载数据并聚类
    print("\n[Step 1] Loading data and clustering...")
    facts_to_edit, demo_corpus = load_data(args.data_path)
    clusters = cluster_facts_by_relation(facts_to_edit)
    
    # 打印聚类统计
    print_cluster_statistics(clusters, top_n=15)
    
    # 保存完整聚类信息
    cluster_info_path = os.path.join(args.output_dir, 'cluster_info.json')
    save_cluster_info(clusters, cluster_info_path)
    
    # Step 2: 过滤 clusters
    print("\n[Step 2] Filtering clusters...")
    clusters_to_evolve = filter_clusters(
        clusters,
        min_size=args.min_cluster_size,
        max_size=args.max_cluster_size,
        top_k=args.top_k_clusters
    )
    
    print(f"After filtering: {len(clusters_to_evolve)} clusters selected for evolution")
    total_facts = sum(c['size'] for c in clusters_to_evolve)
    print(f"Total facts to evolve: {total_facts}")
    
    if len(clusters_to_evolve) == 0:
        print("No clusters to evolve! Adjust filtering parameters.")
        return
    
    # 保存选中的 clusters 信息
    selected_info = {
        'num_selected': len(clusters_to_evolve),
        'total_facts': total_facts,
        'clusters': [
            {
                'relation_id': c['relation_id'],
                'size': c['size']
            } for c in clusters_to_evolve
        ]
    }
    with open(os.path.join(args.output_dir, 'selected_clusters.json'), 'w') as f:
        json.dump(selected_info, f, indent=2)
    
    # Step 3: 对每个 cluster 进行进化
    print("\n[Step 3] Starting clustered evolution...")
    all_results = evo_agent_main_clustered(args, clusters_to_evolve)
    
    # Step 4: 汇总结果
    print("\n[Step 4] Saving summary...")
    summary = {
        'timestamp': timestamp,
        'parameters': vars(args),
        'num_clusters_evolved': len(all_results),
        'total_time_seconds': time.time() - start_time,
        'cluster_results': {}
    }
    
    for relation_id, result in all_results.items():
        summary['cluster_results'][relation_id] = {
            'num_facts': result['cluster_info']['num_facts'],
            'pareto_front_size': len(result['pareto_agents']),
            'best_fitness': {
                'efficacy': max(f[0] for f in result['pareto_fitness']),
                'generalization': max(f[1] for f in result['pareto_fitness']),
                'specificity': max(f[2] for f in result['pareto_fitness'])
            },
            'avg_pareto_fitness': {
                'efficacy': sum(f[0] for f in result['pareto_fitness']) / len(result['pareto_fitness']),
                'generalization': sum(f[1] for f in result['pareto_fitness']) / len(result['pareto_fitness']),
                'specificity': sum(f[2] for f in result['pareto_fitness']) / len(result['pareto_fitness'])
            }
        }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    
    # 打印最终统计
    print("\n" + "="*70)
    print("Evolution Summary")
    print("="*70)
    print(f"Total clusters evolved: {len(all_results)}/{len(clusters_to_evolve)}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Results directory: {args.output_dir}")
    print("="*70)
    
    # 打印每个 cluster 的最佳结果
    print("\nBest fitness per cluster:")
    print(f"{'Relation ID':<15} {'Facts':<8} {'Pareto Size':<12} {'Best ES':<10} {'Best PS':<10} {'Best NS':<10}")
    print("-"*70)
    for relation_id, info in summary['cluster_results'].items():
        print(f"{relation_id:<15} {info['num_facts']:<8} {info['pareto_front_size']:<12} "
              f"{info['best_fitness']['efficacy']:<10.4f} "
              f"{info['best_fitness']['generalization']:<10.4f} "
              f"{info['best_fitness']['specificity']:<10.4f}")
    print("="*70)


if __name__ == '__main__':
    # 设置代理（如果需要）
    proxy = os.getenv("HTTPS_PROXY")
    if proxy:
        print(f"Using proxy: {proxy}")
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["HTTP_PROXY"] = proxy
    
    main()
