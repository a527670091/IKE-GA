# Evo-Agent Phase 2: Agent 协作与自我改进

## 🎯 Phase 2 核心目标

将传统的 **交叉（Crossover）** 和 **变异（Mutation）** 算子替换为 **Agent 的主动行为**：

1. **`Agent.collaborate()`**：两个 Agent 基于自身 fitness 和上下文，主动决定如何协作生成后代
2. **`Agent.self_improve()`**：Agent 基于反馈信息，主动反思并改进自己的策略

---

## 📊 Phase 1 vs Phase 2 对比

| 维度 | Phase 1 | Phase 2 |
|------|---------|---------|
| **交叉** | `crossover(parent1, parent2)` | `parent1.collaborate(parent2, context)` |
| **变异** | `mutation(individual)` | `agent.self_improve(feedback, context)` |
| **Agent 感知** | ❌ Agent 不知道自己的 fitness | ✅ Agent 能感知自己和他人的 fitness |
| **上下文意识** | ❌ 无关系类型感知 | ✅ Agent 知道自己在优化哪类关系 |
| **演示质量** | ⚠️ LLM 盲目生成 | ✅ Agent 主动确保 Prompt-Answer 对应 |
| **目标导向** | ❌ 无明确优化方向 | ✅ Agent 针对弱项改进 |

---

## 🔧 核心实现

### 1. Agent 类扩展

```python
class Agent:
    # Phase 1 属性
    strategy: list[str]
    fitness: tuple[float, float, float]
    agent_id: int
    
    # Phase 2 新增方法
    def collaborate(self, other: Agent, context: dict, evo_model_name: str, k_demos: int) -> Agent:
        """
        两个 Agent 协作生成后代。
        
        context 包含:
        - relation_id: 当前优化的关系类型（如 P413, P176）
        - cluster_size: cluster 中的事实数量
        - generation: 当前进化代数
        """
        
    def self_improve(self, feedback: dict, context: dict, evo_model_name: str, k_demos: int) -> Agent:
        """
        Agent 基于反馈自我改进。
        
        feedback 包含:
        - current_fitness: 当前适应度
        - avg_fitness: 种群平均适应度
        - best_fitness: 种群最佳适应度
        """
```

---

### 2. 协作 Prompt 设计

```python
prompt = f"""你是一个智能教学助手（Agent），现在需要与另一个 Agent 协作。

【任务背景】
关系类型: {context['relation_id']}
数据规模: {context['cluster_size']} 条事实

【你的表现】
适应度: ES={self.fitness[0]:.3f}, PS={self.fitness[1]:.3f}, NS={self.fitness[2]:.3f}
你的演示集合:
{self_strategy_str}

【协作伙伴的表现】
适应度: ES={other.fitness[0]:.3f}, PS={other.fitness[1]:.3f}, NS={other.fitness[2]:.3f}
协作伙伴的演示集合:
{other_strategy_str}

【协作要求】
1. 分析你和协作伙伴的优劣势
2. 特别关注演示质量：
   - 演示中的 "Prompt" 和 "Answer" 必须对应
   - 避免混乱或自相矛盾的演示
3. 融合双方优点，生成 {k_demos} 条新演示
4. 如果 NS（特异性）较低，优先改进演示质量

【输出格式】
直接输出 {k_demos} 条演示，用 --- 分隔...
"""
```

**关键特性**：
- ✅ Agent 能看到双方的 fitness
- ✅ Agent 知道当前优化的关系类型
- ✅ 明确要求确保 Prompt-Answer 对应（解决 Phase 1 的演示质量问题）
- ✅ 针对特异性低的情况给出优化建议

---

### 3. 自我改进 Prompt 设计

```python
prompt = f"""你是一个智能教学助手（Agent），需要基于反馈进行自我改进。

【你当前的表现】
你的适应度: {current_str}
种群平均: {avg_str}
种群最佳: {best_str}
你的弱项: {weakness_str}

【你当前的演示集合】
{strategy_str}

【自我改进要求】
1. 反思你的演示存在的问题：
   - "Prompt" 和 "Answer" 是否对应？
   - 演示是否清晰、一致、无矛盾？
2. 如何改进才能提升弱项指标？
3. 生成改进后的 {k_demos} 条演示
"""
```

**关键特性**：
- ✅ Agent 知道自己哪些指标弱
- ✅ Agent 与种群平均和最佳对比
- ✅ 引导 Agent 反思演示质量

---

### 4. 进化循环修改

#### Phase 1（旧）

```python
for generation in range(num_generations):
    # 选择
    parent_a, parent_b = selection(agents)
    
    # 交叉
    child_strategy = crossover(parent_a.strategy, parent_b.strategy, llm, k)
    
    # 变异
    child_strategy = mutation(child_strategy, llm, corpus, k)
    
    # 创建后代
    child = Agent(strategy=child_strategy)
```

#### Phase 2（新）

```python
for generation in range(num_generations):
    # 构造上下文
    context = {
        'relation_id': 'P413',
        'cluster_size': 95,
        'generation': generation + 1
    }
    
    # 选择
    parent_a, parent_b = selection(agents)
    
    # Agent 协作
    child = parent_a.collaborate(parent_b, context, llm, k)
    
    # Agent 自我改进
    feedback = {
        'current_fitness': child.fitness,
        'avg_fitness': avg(agents),
        'best_fitness': best(agents)
    }
    child = child.self_improve(feedback, context, llm, k)
```

---

## 🚀 使用方式

### 测试 Phase 2

**小规模测试**（验证功能）：

```bash
CUDA_VISIBLE_DEVICES='5,6' \
python run_evo_agent_clustered.py \
  --top_k_clusters 2 \
  --population_size 4 \
  --num_generations 2 \
  --output_dir ./phase2_test \
  --target_model_name EleutherAI/gpt-neo-1.3B \
  --evo_model_name gemini-2.5-pro \
  --gpu_devices 0,1
```

**中等规模实验**（对比 Phase 1）：

```bash
CUDA_VISIBLE_DEVICES='5,6' \
python run_evo_agent_clustered.py \
  --top_k_clusters 5 \
  --min_cluster_size 20 \
  --population_size 6 \
  --num_generations 5 \
  --output_dir ./phase2_medium \
  --target_model_name EleutherAI/gpt-neo-1.3B \
  --evo_model_name gemini-2.5-pro \
  --gpu_devices 0,1
```

---

## 🎯 预期改进

基于 Phase 1 测试结果的分析，Phase 2 应该能显著改进：

### 问题 1：P413 的低特异性（NS=0.12）

**Phase 1 问题**：
```
New Fact: Pat Swilling is goaltender.
Prompt: Doug Buffone plays as
A: linebacker
```
→ Prompt 和 Answer 不对应，导致演示混乱

**Phase 2 改进**：
- ✅ Agent 被明确要求确保 Prompt-Answer 对应
- ✅ Agent 能看到 NS 很低，会优先改进演示质量
- ✅ Agent 反思弱项时，会意识到演示不一致的问题

### 问题 2：演示质量参差不齐

**Phase 1 问题**：
- 盲目交叉可能产生自相矛盾的演示
- 变异没有明确目标

**Phase 2 改进**：
- ✅ 协作时，Agent 分析双方优劣，保留高质量部分
- ✅ 自我改进时，Agent 针对弱项改进

---

## 📈 对比实验建议

### 实验设计

1. **运行 Phase 2**（当前代码）：
   ```bash
   python run_evo_agent_clustered.py --top_k_clusters 5 --output_dir ./phase2_results
   ```

2. **对比 Phase 1 和 Phase 2**：

| Cluster | Phase 1 NS | Phase 2 NS | 提升 |
|---------|-----------|-----------|------|
| P413 | 0.12 | ? | ? |
| P176 | 0.42 | ? | ? |
| ... | ... | ... | ... |

### 关键指标

- **特异性提升**：P413 的 NS 是否能从 0.12 提升到 > 0.20？
- **演示质量**：生成的演示中 Prompt-Answer 对应率
- **收敛速度**：是否能更快达到好的解？

---

## 🔍 调试和优化

### 查看 Agent 生成的演示

```bash
# 查看某个 cluster 的帕累托前沿
cat phase2_test/cluster_P413/pareto_front.json | python -m json.tool

# 手动检查演示质量
# 看 Prompt 和 Answer 是否对应
```

### 如果演示质量仍然差

**可能原因**：
1. LLM 模型理解能力不足
2. Prompt 设计需要优化

**解决方案**：
1. 换更强的 LLM（如 `gemini-2.5-pro` → `gpt-4`）
2. 在 Prompt 中加入更多示例
3. 加入后处理验证（检查 Prompt-Answer 对应性）

---

## 🎓 理论意义

Phase 2 的实现体现了**具身智能**（Embodied AI）的核心思想：

1. **感知（Perception）**：Agent 能感知自己的 fitness 和环境上下文
2. **决策（Decision）**：Agent 主动决定如何协作和改进
3. **行动（Action）**：Agent 生成改进后的演示
4. **学习（Learning）**：通过多代进化，Agent 不断优化策略

这种设计更接近**真实的智能系统**，而非传统遗传算法的盲目搜索。

---

## 📝 后续工作

### Phase 3 可能方向

1. **Agent 记忆**：让 Agent 记住之前失败的尝试
2. **Agent 社交**：引入"导师-学生"关系，优秀 Agent 指导弱 Agent
3. **多智能体博弈**：Agent 之间竞争资源（演示池）
4. **元学习**：Agent 学习如何更好地协作和改进

---

## 🎉 总结

Phase 2 的核心价值：

✅ **从"盲目搜索"到"智能决策"**  
✅ **从"无目标变异"到"有针对性改进"**  
✅ **从"演示质量参差"到"主动质量控制"**  

立即测试 Phase 2，看看能否显著提升 P413 的特异性！

```bash
CUDA_VISIBLE_DEVICES='5,6' \
python run_evo_agent_clustered.py \
  --top_k_clusters 2 \
  --population_size 4 \
  --num_generations 3 \
  --output_dir ./phase2_first_run \
  --gpu_devices 0,1
```
