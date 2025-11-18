"""
聚类工具模块

提供基于 relation_id 的事实聚类功能，用于分组进化。
"""

from collections import defaultdict
from typing import List, Dict, Any
import json


def cluster_facts_by_relation(facts_to_edit: List[Dict]) -> List[Dict[str, Any]]:
    """
    按 relation_id 对事实进行聚类。
    
    Args:
        facts_to_edit: 事实列表，每个事实是一个字典
    
    Returns:
        cluster_list: 聚类结果列表，每个元素格式为:
        {
            'relation_id': str,     # 关系ID（如 'P108'）
            'size': int,            # 该cluster包含的事实数量
            'facts': [              # 事实列表
                {
                    'fact_idx': int,      # 原始索引
                    'fact_data': dict     # 事实数据
                },
                ...
            ]
        }
    """
    clusters_dict = defaultdict(list)
    
    # 按 relation_id 分组
    for idx, fact in enumerate(facts_to_edit):
        relation = fact['requested_rewrite']['relation_id']
        clusters_dict[relation].append({
            'fact_idx': idx,
            'fact_data': fact
        })
    
    # 转为列表格式
    cluster_list = []
    for relation_id, facts in clusters_dict.items():
        cluster_list.append({
            'relation_id': relation_id,
            'size': len(facts),
            'facts': facts
        })
    
    # 按大小降序排序（可选，方便后续处理）
    cluster_list.sort(key=lambda x: x['size'], reverse=True)
    
    return cluster_list


def print_cluster_statistics(clusters: List[Dict[str, Any]], top_n: int = 10):
    """
    打印聚类统计信息。
    
    Args:
        clusters: 聚类结果
        top_n: 显示前 N 个最大的 cluster
    """
    print(f"\n{'='*70}")
    print(f"聚类统计")
    print(f"{'='*70}")
    print(f"总 cluster 数: {len(clusters)}")
    print(f"总事实数: {sum(c['size'] for c in clusters)}")
    print(f"\n前 {top_n} 个最大的 cluster:")
    print(f"{'-'*70}")
    print(f"{'Rank':<6} {'Relation ID':<15} {'Size':<10} {'Percentage':<12}")
    print(f"{'-'*70}")
    
    total_facts = sum(c['size'] for c in clusters)
    for i, cluster in enumerate(clusters[:top_n]):
        percentage = (cluster['size'] / total_facts) * 100
        print(f"{i+1:<6} {cluster['relation_id']:<15} {cluster['size']:<10} {percentage:.1f}%")
    
    print(f"{'-'*70}")
    
    # 统计小 cluster
    small_clusters = [c for c in clusters if c['size'] < 10]
    if small_clusters:
        print(f"\n小 cluster (size < 10): {len(small_clusters)} 个")
        print(f"包含事实总数: {sum(c['size'] for c in small_clusters)}")
    
    print(f"{'='*70}\n")


def filter_clusters(clusters: List[Dict[str, Any]], 
                   min_size: int = 1, 
                   max_size: int = None,
                   top_k: int = None) -> List[Dict[str, Any]]:
    """
    过滤聚类，只保留符合条件的 cluster。
    
    Args:
        clusters: 聚类结果
        min_size: 最小 cluster 大小（默认1，即不过滤）
        max_size: 最大 cluster 大小（默认None，即不限制）
        top_k: 只保留前 k 个最大的 cluster（默认None，即不限制）
    
    Returns:
        filtered_clusters: 过滤后的聚类列表
    """
    filtered = clusters
    
    # 按大小过滤
    if min_size > 1:
        filtered = [c for c in filtered if c['size'] >= min_size]
    if max_size is not None:
        filtered = [c for c in filtered if c['size'] <= max_size]
    
    # 按数量过滤
    if top_k is not None:
        filtered = filtered[:top_k]
    
    return filtered


def save_cluster_info(clusters: List[Dict[str, Any]], output_path: str):
    """
    保存聚类信息到文件。
    
    Args:
        clusters: 聚类结果
        output_path: 输出文件路径
    """
    # 简化格式，只保存关键信息
    simplified_clusters = []
    for cluster in clusters:
        simplified_clusters.append({
            'relation_id': cluster['relation_id'],
            'size': cluster['size'],
            'fact_indices': [f['fact_idx'] for f in cluster['facts']]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_clusters': len(clusters),
            'total_facts': sum(c['size'] for c in clusters),
            'clusters': simplified_clusters
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[Clustering] Cluster info saved to {output_path}")


def load_cluster_info(input_path: str) -> Dict:
    """
    从文件加载聚类信息。
    
    Args:
        input_path: 输入文件路径
    
    Returns:
        cluster_info: 聚类信息字典
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)
