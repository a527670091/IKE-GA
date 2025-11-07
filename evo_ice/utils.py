import os
# 设置Hugging Face国内镜像，加速模型下载
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 删除此处的硬编码 ---
# 我们不应该在代码内部硬编码指定使用哪些GPU。
# 硬件的选择应该完全通过启动脚本时的环境变量（如 CUDA_VISIBLE_DEVICES）来控制。
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, BatchEncoding
from typing import List, Tuple
import json
import numpy as np

# --- 全局常量 ---
COUNTERFACT_SPLIT_INDEX = 2000

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_name: str, device: str):
    """
    加载指定的预训练语言模型和分词器到特定设备。
    """
    print(f"Loading model: {model_name} to device: {device}...")
    
    # 根据设备选择更合适的精度，GPU上使用float16以减少显存占用
    dtype = torch.float16 if str(device).startswith('cuda') else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # 为decoder-only模型（如GPT-J）设置padding_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Model loaded successfully on {device}.")
    return model, tokenizer

def load_data(counterfact_path: str = './counterfact.json'):
    """
    加载COUNTERFACT数据集。
    """
    print(f"Loading data from {counterfact_path}...")
    with open(counterfact_path, 'r') as f:
        lines = json.load(f)
    
    # 将数据集分为待编辑的事实和用于构建演示的语料库
    facts_to_edit = lines[:COUNTERFACT_SPLIT_INDEX]
    demo_corpus = lines[COUNTERFACT_SPLIT_INDEX:]
    print(f"Data loaded: {len(facts_to_edit)} facts to edit, {len(demo_corpus)} examples in demo corpus.")
    return facts_to_edit, demo_corpus

def calculate_fitness(
    individual: list[str], 
    fact_to_edit: dict, 
    model, 
    tokenizer
) -> tuple[float, float, float]:
    """
    计算单个个体（演示上下文）的适应度向量。
    该版本经过优化，将所有评估合并为一次批量调用。
    """
    
    # 1. 提取事实信息
    rewrite_info = fact_to_edit['requested_rewrite']
    prompt_template = rewrite_info['prompt']
    subject = rewrite_info['subject']
    target_new = rewrite_info['target_new']['str']
    target_true = rewrite_info['target_true']['str']
    
    prompt = prompt_template.format(subject)
    
    paraphrase_prompts = fact_to_edit.get('paraphrase_prompts', [])
    neighborhood_prompts = fact_to_edit.get('neighborhood_prompts', [])

    # --- 批量化评估 ---
    # 2. 将所有需要评估的查询收集到一个大批次中
    all_prompts = []
    all_targets = [] # -> List[Tuple(correct_target, incorrect_target)]

    # 效力 (Efficacy)
    all_prompts.append(prompt)
    all_targets.append((target_new, target_true))
    efficacy_slice = slice(0, len(all_prompts))

    # 泛化性 (Generalization / Paraphrase)
    paraphrase_start_idx = len(all_prompts)
    for para_prompt in paraphrase_prompts:
        all_prompts.append(para_prompt)
        all_targets.append((target_new, target_true))
    paraphrase_slice = slice(paraphrase_start_idx, len(all_prompts))

    # 特异性 (Specificity / Neighborhood)
    neighborhood_start_idx = len(all_prompts)
    for neighbor_prompt in neighborhood_prompts:
        all_prompts.append(neighbor_prompt)
        # 对于特异性，正确答案是原始答案 target_true
        all_targets.append((target_true, target_new))
    neighborhood_slice = slice(neighborhood_start_idx, len(all_prompts))
    
    # 如果没有任何要评估的 prompts，直接返回满分
    if not all_prompts:
        return (1.0, 1.0, 1.0)
    
    # 3. 一次性进行批量评估
    base_query_prefix = f'New Fact: {prompt} {target_new}\nPrompt: '
    # Efficacy + Paraphrase 一起评估，targets 顺序为 [target_new, target_true]
    ep_prompts = [prompt] + paraphrase_prompts
    ep_prefixes = [f"{base_query_prefix}{p}" for p in ep_prompts]
    ep_ppls = _icl_lm_eval_batched(model, tokenizer, individual, ep_prefixes, [target_new, target_true])

    # Neighborhood 单独评估，targets 顺序为 [target_true, target_new]
    nb_prefixes = [f"{base_query_prefix}{p}" for p in neighborhood_prompts]
    nb_ppls = _icl_lm_eval_batched(model, tokenizer, individual, nb_prefixes, [target_true, target_new])

    # 4. 解析结果并计算各个维度的分数
    # Efficacy: ep_ppls 的第一个元素
    efficacy_correct = 0
    if ep_ppls:
        if ep_ppls[0][0] < ep_ppls[0][1]:
            efficacy_correct = 1
    efficacy_total = 1 if ep_ppls else 0

    # Paraphrase: ep_ppls 的后续元素
    para_correct = 0
    para_total = max(len(ep_ppls) - 1, 0)
    if para_total > 0:
        for i in range(1, len(ep_ppls)):
            if ep_ppls[i][0] < ep_ppls[i][1]:
                para_correct += 1

    # Neighborhood: 使用 nb_ppls
    neighbor_correct = 0
    neighbor_total = len(nb_ppls)
    for i in range(len(nb_ppls)):
        if nb_ppls[i][0] < nb_ppls[i][1]:
            neighbor_correct += 1

    efficacy_score = efficacy_correct / efficacy_total if efficacy_total > 0 else 1.0
    generalization_score = para_correct / para_total if para_total > 0 else 1.0
    specificity_score = neighbor_correct / neighbor_total if neighbor_total > 0 else 1.0
    
    return (efficacy_score, generalization_score, specificity_score)


def load_corpus_indices(path: str = 'corpus_idx.txt'):
    """加载预先计算好的k-NN演示语料库索引。"""
    print(f"Loading corpus indices from {path}...")
    with open(path, 'r') as fIn:
        lines = fIn.readlines()
    lines = [line.strip() for line in lines]
    corpus_idx = [[int(idx) for idx in line.split()] for line in lines]
    print("Corpus indices loaded.")
    return corpus_idx

# ------------------------------------------------------------------
# 多目标进化算法 (NSGA-II) 核心辅助函数
# ------------------------------------------------------------------

def dominates(p_fitness: tuple, q_fitness: tuple) -> bool:
    """
    判断个体p是否支配个体q。
    """
    all_not_worse = all(p_f >= q_f for p_f, q_f in zip(p_fitness, q_fitness))
    one_strictly_better = any(p_f > q_f for p_f, q_f in zip(p_fitness, q_fitness))
    return all_not_worse and one_strictly_better

def non_dominated_sort(population: list, fitness_scores: list[tuple]) -> list[list[int]]:
    """
    对种群进行非支配排序。
    """
    S = [[] for _ in range(len(population))]
    fronts = [[]]
    n = [0] * len(population)
    rank = [-1] * len(population)

    for p_idx, p_fitness in enumerate(fitness_scores):
        for q_idx, q_fitness in enumerate(fitness_scores):
            if p_idx == q_idx:
                continue
            if dominates(p_fitness, q_fitness):
                S[p_idx].append(q_idx)
            elif dominates(q_fitness, p_fitness):
                n[p_idx] += 1
        
        if n[p_idx] == 0:
            rank[p_idx] = 0
            fronts[0].append(p_idx)

    i = 0
    while fronts[i]:
        Q = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    rank[q_idx] = i + 1
                    Q.append(q_idx)
        i += 1
        fronts.append(Q)
    
    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts


def calculate_crowding_distance(front_indices: list[int], fitness_scores: list[tuple]):
    """
    计算一个前沿内所有个体的拥挤度。
    """
    num_individuals = len(front_indices)
    if num_individuals == 0:
        return {}
        
    distances = {idx: 0.0 for idx in front_indices}
    num_objectives = len(fitness_scores[0])

    for m in range(num_objectives):
        sorted_indices = sorted(front_indices, key=lambda idx: fitness_scores[idx][m])
        
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        if num_individuals > 2:
            f_max = fitness_scores[sorted_indices[-1]][m]
            f_min = fitness_scores[sorted_indices[0]][m]
            
            if f_max == f_min:
                continue

            for i in range(1, num_individuals - 1):
                distances[sorted_indices[i]] += (fitness_scores[sorted_indices[i+1]][m] - fitness_scores[sorted_indices[i-1]][m]) / (f_max - f_min)
    
    return distances


def _icl_lm_eval_batched(
    model, 
    tokenizer, 
    icl_examples: list[str], 
    prompts: list[str], 
    targets: list[str]
) -> list[list[float]]:
    """
    对一批prompts进行评估，返回每个prompt对应的targets的困惑度列表。
    """
    if not prompts:
        return []
        
    # 为decoder-only模型设置左填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # 准备批处理数据
    # 对于每个prompt，我们都需要和每个target组合
    prompt_prefixes = ["".join(icl_examples) + p for p in prompts]
    full_texts = [prefix + " " + target for prefix in prompt_prefixes for target in targets]
    
    # 批量编码
    encodings = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    input_ids = encodings['input_ids']
    
    # --- 关键修复：将输入数据移动到模型所在的设备 ---
    # 无论模型在单卡还是多卡，model.device都会指向模型第一层所在的设备。
    # 我们需要确保输入数据在同一个设备上。
    input_ids = input_ids.to(model.device)

    # 准备labels，同时处理好masking
    labels = input_ids.clone()
    # 获取每个prompt部分的长度，用于后续mask
    # --- 性能修复 ---
    # 旧方法（非常慢）: prompt_lengths = [len(tokenizer.encode(p)) for p in prompt_prefixes]
    # 新方法（非常快）: 使用分词器对所有前缀进行一次批处理编码，然后获取长度。
    prompt_prefix_encodings = tokenizer(prompt_prefixes, padding=False, truncation=False)
    prompt_lengths = [len(e) for e in prompt_prefix_encodings['input_ids']]
    
    for i in range(len(full_texts)):
        prompt_idx = i // len(targets) # 当前文本属于哪个prompt
        prompt_len = prompt_lengths[prompt_idx]
        
        # input_ids中实际内容的开始位置 (跳过padding)
        actual_content_start = (input_ids[i] != tokenizer.pad_token_id).nonzero(as_tuple=True)[0].min().item()
        
        # 需要mask掉的长度 = prompt的长度
        mask_len = actual_content_start + prompt_len
        if mask_len < labels.shape[1]:
            labels[i, :mask_len] = -100

    # 将padding部分的label也设置为-100
    labels[labels == tokenizer.pad_token_id] = -100

    # 模型推理和损失计算
    with torch.no_grad():
        # --- 最终修复方案：手动计算损失 ---
        # 1. 只传入input_ids，获取模型输出的logits
        outputs = model(input_ids)
        logits = outputs.logits
        
        # 2. 手动进行shift操作，这与HuggingFace内部的逻辑一致
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 3. 将labels移动到与logits完全相同的设备上
        shift_labels = shift_labels.to(shift_logits.device)
        
        # 4. 手动计算损失，并设置reduction='none'以获取每个样本的loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 5. 将loss重新塑形并计算每个样本的PPL
        loss_per_token = loss.view(shift_logits.size(0), -1)
        # 忽略被mask掉的token (-100)
        actual_tokens_per_sample = (shift_labels != -100).sum(dim=1)
        # 避免除以零
        actual_tokens_per_sample[actual_tokens_per_sample == 0] = 1 
        
        mean_loss_per_sample = loss_per_token.sum(dim=1) / actual_tokens_per_sample
        ppls = torch.exp(mean_loss_per_sample)

    # 将扁平的PPL列表重塑为 [num_prompts, num_targets]
    ppls = ppls.view(len(prompts), len(targets)).tolist()
    
    return ppls


def icl_lm_eval(model, tokenizer, icl_examples, targets, x):
    """
    从icl.py迁移过来的核心评估函数，计算困惑度。
    """
    ppls = [] 
    for target in targets:
        # 添加一个空格以确保分词正确
        target_text = ' ' + target
        tgt_len = len(tokenizer.encode(target_text))
        
        full_text = ''.join(icl_examples) + x + target_text
        encodings = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=1024)
        
        # 当使用 device_map="auto" 时，我们不应手动将输入移动到特定设备。
        # accelerate 会自动处理设备放置。
        input_ids = encodings['input_ids']
        target_ids = input_ids.clone()
        
        seq_len = input_ids.shape[1]
        
        # 确保tgt_len不会溢出
        if tgt_len >= seq_len:
            # 这种情况通常不应该发生，但作为安全检查
            ppl = float('inf')
        else:
            target_ids[:, :-tgt_len] = -100
            with torch.no_grad():
                # --- 最终修复方案：手动计算损失 ---
                # 1. 只传入input_ids，获取模型输出的logits
                outputs = model(input_ids)
                logits = outputs.logits

                # 2. 手动进行shift操作，这与HuggingFace内部的逻辑一致
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                
                # 3. 将labels移动到与logits完全相同的设备上
                shift_labels = shift_labels.to(shift_logits.device)

                # 4. 手动计算损失
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                ppl = torch.exp(loss).item()

        ppls.append(ppl)
    return ppls
