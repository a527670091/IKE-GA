"""
Evo-Agent: Agent-based Evolutionary In-Context Editing Framework
Phase 1: Agent abstraction with equivalent behavior to Evo-ICE

支持两种运行模式：
1. 单事实模式：evo_agent_main - 对单个事实进行进化
2. 聚类模式：evo_agent_main_clustered - 对多个聚类后的事实分组进化
"""

from .agents import Agent
from .evo_agent_ice import evo_agent_main, evo_agent_main_clustered
from .clustering import cluster_facts_by_relation, filter_clusters, print_cluster_statistics

__all__ = [
    'Agent', 
    'evo_agent_main',
    'evo_agent_main_clustered',
    'cluster_facts_by_relation',
    'filter_clusters',
    'print_cluster_statistics'
]
