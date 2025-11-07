import os
import re
import random
from openai import OpenAI
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

# --- 自动加载 .env 文件中的环境变量 ---
# 这行代码会自动寻找项目根目录下的 .env 文件并加载它
# 这样我们就可以安全地在 .env 文件中管理 API 密钥，而无需硬编码
load_dotenv()

# --- LLM API 调用模块 ---

def call_evo_llm(prompt: str, model_name: str) -> Optional[str]:
    """
    智能调度器，根据环境变量决定使用哪个LLM服务。
    优先级: 中转API > Google Gemini > OpenAI
    """
    # 优先检查是否存在中转API的配置
    hiapi_base_url = os.getenv("HIAPI_BASE_URL")
    hiapi_api_key = os.getenv("HIAPI_API_KEY")

    if hiapi_base_url and hiapi_api_key:
        print(f"检测到中转API配置，将通过 {hiapi_base_url} 调用模型...")
        return _call_openai_compatible_proxy(prompt, model_name, hiapi_base_url, hiapi_api_key)
    
    # 如果没有中转API配置，则回退到原来的逻辑
    if 'gemini' in model_name.lower():
        print("未检测到中转API，回退到原生Gemini API调用...")
        return _call_gemini_api(prompt, model_name)
    else:
        print("未检测到中转API，回退到原生OpenAI API调用...")
        return _call_openai_api(prompt, model_name)

def _call_openai_compatible_proxy(prompt: str, model_name: str, base_url: str, api_key: str) -> Optional[str]:
    """通过类OpenAI接口的中转服务调用模型。"""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一位专注于优化AI教学示例的专家，严格遵循指令。"},
                {"role": "user", "content": prompt}
            ]
            # 移除 temperature 和 max_tokens 参数，以匹配用户成功的测试脚本
            # temperature=0.7,
            # max_tokens=2048,
        )
        
        # 增强调试：检查API是否返回了空内容，并打印finish_reason
        content = response.choices[0].message.content if response.choices and response.choices[0].message else None
        
        if not content:
            print("\n------ [调试] API 返回了空内容! ------")
            finish_reason = response.choices[0].finish_reason if response.choices else "未知"
            print(f"-> 完成原因 (Finish Reason): {finish_reason}")
            print("-> 完整的 Choice 对象:", response.choices[0] if response.choices else "没有 Choice 对象")
            print("------------------------------------------\n")
            return "" # 返回空字符串以匹配之前的行为

        return content

    except Exception as e:
        print(f"调用中转API时发生错误: {e}")
        return None

def _call_openai_api(prompt: str, model_name: str) -> Optional[str]:
    """调用 OpenAI API。"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("错误：请设置 OPENAI_API_KEY 环境变量以使用OpenAI模型。")
            return None

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一位专注于优化AI教学示例的专家，严格遵循指令。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用 OpenAI API 时发生错误: {e}")
        return None

def _call_gemini_api(prompt: str, model_name: str) -> Optional[str]:
    """调用 Google Gemini API。"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("错误：请设置 GOOGLE_API_KEY 环境变量以使用Gemini模型。")
            return None
        
        genai.configure(api_key=api_key)

        # --- 代理解决方案：自动适应版本 ---
        transport = None
        use_transport = False
        proxy_url = os.getenv("HTTPS_PROXY")
        if proxy_url:
            print(f"检测到代理设置 ({proxy_url})，将尝试通过代理连接...")
            try:
                # 优先尝试新版本支持的 transport 方式
                transport = genai.transport.RESTTransport(proxy=proxy_url)
                use_transport = True
                print("已使用 'transport' (新版) 方式配置代理。")
            except AttributeError:
                # 如果 transport 模块不存在，说明是旧版本，回退到依赖环境变量
                print("警告：当前 'google-generativeai' 版本较旧，无 'transport' 模块。")
                print("将回退到标准环境变量代理方式，请确保已正确设置 HTTPS_PROXY。")
        else:
            print("未检测到代理设置，将尝试直接连接...")
        # --- 方案结束 ---

        # 为Gemini设置安全配置
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # --- 关键修复 ---
        # 只有在新版本（use_transport 为 True）的情况下，才传递 transport 参数
        if use_transport:
            model = genai.GenerativeModel(model_name, transport=transport)
        else:
            model = genai.GenerativeModel(model_name)
        # --- 修复结束 ---
        
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.7
            )
        )
        return response.text
    except Exception as e:
        print(f"调用 Google Gemini API 时发生错误: {e}")
        return None


# --- Prompt 构造模块 ---

def format_individual_for_prompt(individual: list[str]) -> str:
    """将演示集格式化为编号列表字符串，以便放入Prompt。"""
    return "\n".join(f"{i+1}. {demo.strip()}" for i, demo in enumerate(individual))

def get_crossover_prompt(parent1_str: str, parent2_str: str) -> str:
    """根据方案文档，构造交叉操作的Prompt。"""
    return f"""你是一位专注于创建和重组教学范例的AI专家。你的任务是严格遵循指令，将两个演示文集（集合A和集合B）的优点融合，创建一个更优秀的后代文集。

**背景**: 这两个文集都旨在教会一个AI模型一个新知识。每个演示都以 "New Fact: ..." 开头。

**集合 A**:
---
{parent1_str}
---

**集合 B**:
---
{parent2_str}
---

**指令**:
1.  **分析与挑选**: 从集合A中挑选出最有助于**泛化**的'update'类型示例；从集合B中挑选出最具挑战性、最能防止**过度编辑**的'retain'类型示例。
2.  **重组与生成**: 智能地组合这些挑选出的示例，并补充必要的'copy'类型示例，确保最终集合的结构完整、逻辑清晰，且总示例数与父代相同。
3.  **严格的输出格式**:
    *   **必须**直接输出最终的、格式化的新集合。
    *   每个演示条目之间**必须**用三个连字符 "---" 作为唯一的分隔符。
    *   **绝对不要**包含任何解释性文字、前言、标题或总结。你的输出应该直接以 "New Fact: ..." 开始。
"""

def get_mutation_prompt(individual_str: str) -> str:
    """根据方案文档，构造变异操作的Prompt（重写类型）。"""
    num_demos = individual_str.count('\n') + 1
    demo_to_mutate = random.randint(1, num_demos)
    
    return f"""你是一位语言润色专家。请对以下演示集中的第 {demo_to_mutate} 个示例进行重写，使其表述更清晰、意图更明确，同时保持其核心功能。

**原始集合**:
---
{individual_str}
---

**指令**:
1.  **精确修改**: 专注于修改第 {demo_to_mutate} 个示例。保持集合中其他所有示例不变。
2.  **严格的输出格式**:
    *   **必须**直接输出修改后的完整集合。
    *   每个演示条目之间**必须**用三个连字符 "---" 作为唯一的分隔符。
    *   **绝对不要**包含任何解释性文字、前言、标题或总结。你的输出应该直接以 "New Fact: ..." 开始。
"""

# --- 响应解析模块 ---

def parse_llm_response(response_text: str, k_demos: int) -> Optional[list[str]]:
    """
    解析EvoLLM的响应，将其转换回演示集列表。

    Args:
        response_text (str): LLM返回的原始文本。
        k_demos (int): 期望的演示数量。

    Returns:
        Optional[list[str]]: 解析成功则返回演示列表，否则返回None。
    """
    if response_text is None:
        return None
        
    # 增强1: 尝试找到第一个 "New Fact:"，并从那里开始处理，以忽略可能的前言。
    start_index = response_text.find("New Fact:")
    if start_index == -1:
        print(f"解析错误：在模型响应中未找到任何 'New Fact:' 开头的演示。")
        # 增加关键调试输出，打印模型返回的原始文本
        print("------ [调试] 模型原始响应 START ------")
        print(response_text)
        print("------ [调试] 模型原始响应 END ------")
        return None
    
    processed_text = response_text[start_index:]
    
    # 增强2: 使用更灵活的正则表达式来切分，允许分隔符前后有更多空白
    # 同时过滤掉因切分产生的空字符串
    demos = [demo.strip() for demo in re.split(r'\s*---\s*', processed_text) if demo.strip()]
    
    # 清理每个demo，确保格式正确
    cleaned_demos = []
    for demo in demos:
        # 不再移除编号，因为我们的prompt已经禁止了它
        cleaned_demos.append(demo + "\n\n") # 恢复IKE使用的格式

    # 验证解析后的数量是否符合预期
    if len(cleaned_demos) == k_demos:
        return cleaned_demos
    else:
        print(f"解析错误：期望得到 {k_demos} 个演示，但解析出了 {len(cleaned_demos)} 个。")
        # 增加调试输出
        print("------ [调试] 模型原始响应 (部分) START ------")
        print(response_text[:500] + "...")
        print("------ [调试] 模型原始响应 (部分) END ------")
        return None
