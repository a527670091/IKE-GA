"""
Agent abstraction for Evo-Agent framework.

Phase 1: Agent is a simple data structure wrapping strategy and fitness.
Phase 2: Agent has active behaviors - collaborate() and self_improve().
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Agent:
    """
    Teaching Agent (教学智能体) in the Evo-Agent framework.
    
    Attributes:
        strategy: The agent's teaching strategy, i.e., a demonstration context.
                  This is a list of k demonstration strings.
        fitness: The agent's fitness vector (ES, PS, NS) if evaluated, else None.
                 ES = Efficacy Score
                 PS = Paraphrase/Generalization Score  
                 NS = Neighborhood/Specificity Score
        agent_id: Optional identifier for logging and tracking
    """
    strategy: list[str]
    fitness: Optional[tuple[float, float, float]] = None
    agent_id: Optional[int] = None
    
    def __repr__(self) -> str:
        fitness_str = f"({self.fitness[0]:.3f}, {self.fitness[1]:.3f}, {self.fitness[2]:.3f})" if self.fitness else "Not evaluated"
        return f"Agent(id={self.agent_id}, fitness={fitness_str}, strategy_len={len(self.strategy)})"
    
    def has_fitness(self) -> bool:
        """Check if this agent has been evaluated."""
        return self.fitness is not None
    
    def get_efficacy(self) -> float:
        """Get Efficacy Score (ES)."""
        if not self.has_fitness():
            raise ValueError("Agent has not been evaluated yet")
        return self.fitness[0]
    
    def get_generalization(self) -> float:
        """Get Generalization/Paraphrase Score (PS)."""
        if not self.has_fitness():
            raise ValueError("Agent has not been evaluated yet")
        return self.fitness[1]
    
    def get_specificity(self) -> float:
        """Get Specificity/Neighborhood Score (NS)."""
        if not self.has_fitness():
            raise ValueError("Agent has not been evaluated yet")
        return self.fitness[2]
    
    # ============================================================
    # Phase 2: Agent Active Behaviors
    # ============================================================
    
    def collaborate(self, other: 'Agent', context: Dict[str, Any], evo_model_name: str, k_demos: int) -> 'Agent':
        """
        Phase 2: Agent 主动协作行为。
        
        两个 Agent 基于自身的 fitness 和 cluster 上下文信息，
        主动决定如何融合各自的策略，生成一个后代 Agent。
        
        Args:
            other: 另一个协作的 Agent
            context: 上下文信息，包含：
                - 'relation_id': 当前 cluster 的关系类型
                - 'cluster_size': cluster 中的事实数量
                - 'generation': 当前进化代数
            evo_model_name: 用于协作的 LLM 模型名称
            k_demos: 生成的演示数量
            
        Returns:
            新的 Agent（后代）
        """
        from .llm_operations import call_evo_llm, parse_llm_response, format_individual_for_prompt
        
        # 构造协作 Prompt
        self_fitness_str = f"ES={self.fitness[0]:.3f}, PS={self.fitness[1]:.3f}, NS={self.fitness[2]:.3f}" if self.has_fitness() else "未评估"
        other_fitness_str = f"ES={other.fitness[0]:.3f}, PS={other.fitness[1]:.3f}, NS={other.fitness[2]:.3f}" if other.has_fitness() else "未评估"
        
        self_strategy_str = format_individual_for_prompt(self.strategy)
        other_strategy_str = format_individual_for_prompt(other.strategy)
        
        prompt = f"""你是一个智能教学助手（Agent），现在需要与另一个 Agent 协作，为知识编辑任务生成更优的演示集合。

【任务背景】
关系类型: {context.get('relation_id', 'Unknown')}
数据规模: {context.get('cluster_size', 'N/A')} 条事实
进化代数: {context.get('generation', 'N/A')}

【你的表现】
适应度: {self_fitness_str}
你的演示集合:
{self_strategy_str}

【协作伙伴的表现】
适应度: {other_fitness_str}
协作伙伴的演示集合:
{other_strategy_str}

【评价指标说明】
- ES (Efficacy): 编辑成功率，越高越好
- PS (Paraphrase/Generalization): 泛化性，在改写的问题上仍然有效，越高越好
- NS (Neighborhood/Specificity): 特异性，不影响相关但不同的知识，越高越好

【协作要求】
1. 分析你和协作伙伴的优劣势，特别关注 NS 是否偏低
2. 演示质量要求：
   - 演示中的 "Prompt" 和 "Answer" 必须严格对应
   - 保持演示的一致性和清晰性
3. **特异性保护（重点）**：
   - 在 {k_demos} 条演示中，必须包含至少 3-4 条"邻域保护演示"
   - 邻域保护演示格式：New Fact 描述的是 Subject A 的新知识，但 Prompt 询问的是 Subject B（相关但不同的对象），Answer 必须是 Subject B 的原始正确答案，而不是 Subject A 的新知识
   - 例如：New Fact: Pat Swilling plays as goaltender. / Prompt: What position does Doug Buffone play? / A: linebacker（保持原值）
   - 这样可以教会模型：只修改目标对象，不要影响其他相关对象
4. 剩余的演示用于保证 ES 和 PS：
   - Copy 类型：直接重复新事实
   - Update 类型：用改写的问法测试新事实
5. 融合双方的优点，生成 {k_demos} 条新的演示

【输出格式】
直接输出 {k_demos} 条演示，每条演示用 --- 分隔，格式如下：
New Fact: [事实陈述]
Prompt: [提示]
A: [答案]

---

New Fact: [事实陈述]
Prompt: [提示]
A: [答案]

（不要输出其他分析内容，只输出演示）
"""
        
        try:
            print("[Evo-Agent] Agent collaboration in progress...")
            response_text = call_evo_llm(prompt, evo_model_name)
            child_strategy = parse_llm_response(response_text, k_demos)
            
            if child_strategy is None:
                print("[Evo-Agent] Collaboration parsing failed, returning self as fallback.")
                return Agent(strategy=self.strategy.copy(), fitness=None)
            
            print("[Evo-Agent] Collaboration successful.")
            return Agent(strategy=child_strategy, fitness=None)
            
        except Exception as e:
            print(f"[Evo-Agent] Collaboration failed: {e}, returning self as fallback.")
            return Agent(strategy=self.strategy.copy(), fitness=None)
    
    def self_improve(self, feedback: Dict[str, Any], context: Dict[str, Any], evo_model_name: str, k_demos: int) -> 'Agent':
        """
        Phase 2: Agent 自我改进行为。
        
        Agent 基于自身的 fitness 反馈和 cluster 上下文，
        主动反思并改进自己的策略。
        
        Args:
            feedback: 反馈信息，包含：
                - 'current_fitness': 当前的适应度
                - 'avg_fitness': 种群的平均适应度
                - 'best_fitness': 种群的最佳适应度
            context: 上下文信息（同 collaborate）
            evo_model_name: 用于自我改进的 LLM 模型名称
            k_demos: 演示数量
            
        Returns:
            改进后的新 Agent
        """
        from .llm_operations import call_evo_llm, parse_llm_response, format_individual_for_prompt
        
        current_fitness = feedback.get('current_fitness', self.fitness)
        avg_fitness = feedback.get('avg_fitness', (0.5, 0.5, 0.5))
        best_fitness = feedback.get('best_fitness', (1.0, 1.0, 1.0))
        
        current_str = f"ES={current_fitness[0]:.3f}, PS={current_fitness[1]:.3f}, NS={current_fitness[2]:.3f}"
        avg_str = f"ES={avg_fitness[0]:.3f}, PS={avg_fitness[1]:.3f}, NS={avg_fitness[2]:.3f}"
        best_str = f"ES={best_fitness[0]:.3f}, PS={best_fitness[1]:.3f}, NS={best_fitness[2]:.3f}"
        
        strategy_str = format_individual_for_prompt(self.strategy)
        
        # 分析弱项
        weaknesses = []
        if current_fitness[0] < avg_fitness[0]:
            weaknesses.append("Efficacy (编辑成功率) 低于平均")
        if current_fitness[1] < avg_fitness[1]:
            weaknesses.append("Generalization (泛化性) 低于平均")
        if current_fitness[2] < avg_fitness[2]:
            weaknesses.append("Specificity (特异性) 低于平均")
        
        weakness_str = "、".join(weaknesses) if weaknesses else "表现良好"
        
        prompt = f"""你是一个智能教学助手（Agent），现在需要基于反馈进行自我改进。

【任务背景】
关系类型: {context.get('relation_id', 'Unknown')}
数据规模: {context.get('cluster_size', 'N/A')} 条事实
进化代数: {context.get('generation', 'N/A')}

【你当前的表现】
你的适应度: {current_str}
种群平均: {avg_str}
种群最佳: {best_str}

你的弱项: {weakness_str}

【你当前的演示集合】
{strategy_str}

【评价指标说明】
- ES (Efficacy): 编辑成功率，越高越好
- PS (Paraphrase/Generalization): 泛化性，在改写的问题上仍然有效，越高越好
- NS (Neighborhood/Specificity): 特异性，不影响相关但不同的知识，越高越好

【自我改进要求】
1. 诊断当前演示的问题：
   - 演示中的 "Prompt" 和 "Answer" 是否严格对应？
   - 如果 NS 低于平均，说明演示可能在"过度泛化"：教会模型把新知识应用到了不该改的对象上
2. **针对 NS 低的核心修复方案**：
   - 如果你的 NS < 0.3，必须在 {k_demos} 条演示中加入至少 4-5 条"邻域保护演示"
   - 如果你的 NS 在 0.3-0.5 之间，加入至少 3 条"邻域保护演示"
   - 邻域保护演示定义：
     * New Fact 描述 Subject A 的新知识
     * Prompt 询问 Subject B（与 A 相关但不同的对象）
     * Answer 必须是 Subject B 的原始正确答案，不是 A 的新知识
     * 例如：New Fact: Pat Swilling plays as goaltender. / Prompt: Doug Buffone plays as / A: linebacker（原值）
   - 这类演示能明确告诉模型："只改目标对象，保护其他对象"
3. 如果 ES 或 PS 低，保留一些 Copy 和 Update 类型的演示来提升这两个指标
4. 生成改进后的 {k_demos} 条演示，确保三个指标平衡

【输出格式】
直接输出 {k_demos} 条改进后的演示，每条用 --- 分隔，格式如下：
New Fact: [事实陈述]
Prompt: [提示]
A: [答案]

---

New Fact: [事实陈述]
Prompt: [提示]
A: [答案]

（不要输出分析内容，只输出演示）
"""
        
        try:
            print("[Evo-Agent] Agent self-improvement in progress...")
            response_text = call_evo_llm(prompt, evo_model_name)
            improved_strategy = parse_llm_response(response_text, k_demos)
            
            if improved_strategy is None:
                print("[Evo-Agent] Self-improvement parsing failed, returning self unchanged.")
                return Agent(strategy=self.strategy.copy(), fitness=None)
            
            print("[Evo-Agent] Self-improvement successful.")
            return Agent(strategy=improved_strategy, fitness=None)
            
        except Exception as e:
            print(f"[Evo-Agent] Self-improvement failed: {e}, returning self unchanged.")
            return Agent(strategy=self.strategy.copy(), fitness=None)
