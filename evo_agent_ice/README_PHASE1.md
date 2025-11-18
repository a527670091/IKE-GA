# Evo-Agent Phase 1 实现说明

## 概述

这是 **Evo-Agent Phase 1** 的实现，将原有的 Evo-ICE 框架从"被动染色体"模式升级为"智能体（Agent）"模式。

## Phase 1 目标

✅ **已完成的改造**：
1. **引入 Agent 抽象** (`agents.py`)
   - 每个个体不再是简单的 `list[str]`，而是一个 `Agent` 对象
   - Agent 包含：
     - `strategy: list[str]` - 演示上下文（教学策略）
     - `fitness: tuple[float, float, float] | None` - 适应度向量 (Efficacy, Generalization, Specificity)
     - `agent_id: int` - 智能体唯一标识符

2. **改造主流程** (`evo_agent_ice.py`)
   - 种群从 `list[list[str]]` 改为 `list[Agent]`
   - 适应度评估后直接写入 `Agent.fitness` 属性
   - 选择、更新种群等操作直接基于 Agent 对象

3. **保持算法等价性**
   - Phase 1 的交叉和变异**仍使用原有的 LLM Prompt**
   - NSGA-II 选择逻辑完全不变
   - 确保与 Evo-ICE 的结果统计上等价

## 文件结构

```
evo_agent_ice/
├── __init__.py           # 模块初始化
├── agents.py             # Agent 数据结构定义
├── evo_agent_ice.py      # 主算法流程（Agent版本）
├── utils.py              # 工具函数（从 evo_ice 复制）
├── llm_operations.py     # LLM 操作（从 evo_ice 复制）
└── README_PHASE1.md      # 本文档
```

## 运行方式

### 方式 1：使用新的运行脚本

```bash
python run_evo_agent.py \
    --target_model_name EleutherAI/gpt-j-6B \
    --evo_model_name gemini-2.5-pro \
    --population_size 6 \
    --num_generations 5 \
    --k_demos 10 \
    --gpu_devices 0
```

### 方式 2：直接运行模块

```bash
python -m evo_agent_ice.evo_agent_ice \
    --target_model_name EleutherAI/gpt-j-6B \
    --evo_model_name gemini-2.5-pro
```

## 与 Evo-ICE 的对比

| 维度 | Evo-ICE | Evo-Agent Phase 1 |
|-----|---------|-------------------|
| 种群表示 | `list[list[str]]` | `list[Agent]` |
| 适应度存储 | 独立的 `fitness_scores` 列表 | `Agent.fitness` 属性 |
| 交叉/变异 | 函数式操作 | 函数式操作（等价） |
| 选择机制 | 基于索引的锦标赛 | 基于 Agent 对象的锦标赛 |
| 输出日志 | `[Evo-ICE]` 前缀 | `[Evo-Agent]` 前缀 |

## Phase 1 验证方法

Phase 1 的核心是**保持算法等价性**。验证方法：

1. **相同参数运行对比**
   ```bash
   # 运行原版 Evo-ICE
   python run_evo_ice.py --seed 42 --population_size 6 --num_generations 3
   
   # 运行 Evo-Agent Phase 1
   python run_evo_agent.py --seed 42 --population_size 6 --num_generations 3
   ```

2. **检查结果**
   - 最终 Pareto 前沿的数量应该接近
   - 适应度分布应该统计上相似
   - 性能曲线趋势应该一致

## 下一步：Phase 2

Phase 2 将引入真正的 Agent 行为：

- ✨ **Collaborate** (协作交叉)：基于双方适应度的协作式 Prompt
- ✨ **Self_Improve** (自我改进)：基于自身策略的反思式变异

这将在 `llm_operations.py` 中新增两个函数：
- `get_collaborate_prompt_for_agents(parent_a: Agent, parent_b: Agent) -> str`
- `get_self_improve_prompt_for_agent(agent: Agent) -> str`

并在 `evo_agent_ice.py` 中调用：
- `agent_collaborate(parent_a, parent_b, evo_model_name, k_demos) -> list[str]`
- `agent_self_improve(child_agent, evo_model_name, k_demos) -> list[str]`

## 注意事项

1. **多进程兼容性**
   - Agent 对象被设计为 dataclass，完全可 pickle
   - 在多进程池中只传递 `agent.strategy`，而不是整个 Agent 对象

2. **内存开销**
   - Agent 相比原来的 `list[str]` 增加了约 100 bytes 的开销（fitness + agent_id）
   - 对于种群大小 N=6，总开销可忽略不计

3. **日志标识**
   - 所有日志都带有 `[Evo-Agent]` 前缀，便于区分

## 贡献者

- Phase 1 设计与实现：基于 `方案v2.md` 中的 Evo-Agent 架构
- 基础框架来源：原 `evo_ice/` 模块
