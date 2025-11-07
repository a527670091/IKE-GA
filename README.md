# IKE - 上下文学习知识编辑
Source code for "Can We Edit Factual Knowledge by In-Context Learning?"

## 📖 项目简介
这个项目研究如何通过上下文学习（In-Context Learning）来编辑大型语言模型中的事实知识。简单来说，就是通过给模型提供一些示例，让它学会新的知识或修正错误的知识。

**🎯 新手推荐**：
- 如果你是**第一次运行**，建议先阅读 [快速开始指南.md](./快速开始指南.md)，里面有更简化的步骤说明。
- 如果你想**深入理解代码**，建议阅读 [代码导读.md](./代码导读.md)，里面有详细的代码解析和核心概念讲解。

## 📋 环境要求

### 硬件要求
- **重要**：运行这个项目需要一张较好的显卡（GPU），至少需要16GB显存
- 如果使用更大的模型（如gpt-neox-20b），需要更多显存

### 软件要求
- Python 3.7 或更高版本
- CUDA（用于GPU加速）

### Python依赖包
```
jsonlines==3.1.0
nltk==3.6.7
numpy==1.22.3
openai==0.25.0
sentence_transformers==2.2.0
spacy==3.2.3
torch==1.11.0
tqdm==4.56.0
transformers==4.24.0
```

## 🚀 完整运行步骤

### 第一步：安装环境

1. **创建Python虚拟环境（推荐）**
```bash
# 在项目目录下运行
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

2. **安装依赖包**

**方式1：一键安装（推荐）**
```bash
pip install -r requirements.txt
```

**方式2：手动安装每个包**
```bash
pip install jsonlines==3.1.0
pip install nltk==3.6.7
pip install numpy==1.22.3
pip install openai==0.25.0
pip install sentence-transformers==2.2.0
pip install spacy==3.2.3
pip install torch==1.11.0
pip install tqdm==4.56.0
pip install transformers==4.24.0
```

**如果下载很慢，可以使用国内镜像：**
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第二步：配置API密钥（如果使用API服务）

如果你需要使用Gemini或OpenAI的API服务，需要配置API密钥：

1. **创建.env文件**
   在项目根目录下创建`.env`文件，添加以下内容：
   ```bash
   # Gemini API密钥
   # 获取地址：https://ai.google.dev/gemini-api/docs/api-key
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # OpenAI API密钥
   # 获取地址：https://platform.openai.com/api-keys
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **测试API连接**
   运行测试脚本检查API是否可以正常访问：
   ```bash
   python test_api_connection.py
   ```
   这个脚本会：
   - 测试Gemini API连接
   - 测试OpenAI API连接
   - 显示详细的测试结果和错误信息

**⚠️ 安全提示**：
- `.env`文件已经在`.gitignore`中，不会被提交到Git仓库
- 不要将API密钥硬编码在代码中
- 不要将API密钥分享给他人

### 第三步：准备数据

1. **下载CounterFact数据集**
```bash
# 使用wget下载（Linux/Mac）
wget https://rome.baulab.info/data/dsets/counterfact.json

# 或者使用浏览器直接访问下面的链接下载，然后放到项目目录下
# https://rome.baulab.info/data/dsets/counterfact.json
```
下载后确保`counterfact.json`文件在项目根目录下。

2. **清理数据**
```bash
python clean_paraphrase.py
```
这一步会清理数据集中的无关前缀，让所有提示词格式统一。

### 第四步：构建示例索引（可选）

项目已经包含了预先构建好的`corpus_idx.txt`文件。如果你想自己重新构建，可以运行：

```bash
# 第1步：为所有事实编码（生成语义向量）
python encode_facts.py

# 第2步：为每个测试样本找到最相似的32个示例
python semantic_search.py
```

**注意**：这两步会需要较长时间（几小时），建议使用已有的`corpus_idx.txt`文件。

### 第五步：运行实验

1. **运行IKE主实验**
```bash
# 使用默认模型（GPT-J-6B）
python icl.py

# 使用其他模型
python icl.py --model_name gpt2-xl
python icl.py --model_name EleutherAI/gpt-neo-1.3B
python icl.py --model_name EleutherAI/gpt-j-6B
python icl.py --model_name EleutherAI/gpt-neox-20b
```

2. **运行PROMPT基线实验**
```bash
python prompt.py
```

## 📊 输出结果说明

运行实验后，程序会每隔10个样本输出一次进度，包括：
- 当前处理的样本数量
- 成功率
- 各项指标的得分

最终会输出完整的评估结果。

## 🔧 常见问题

### 问题1：显存不足（CUDA out of memory）
**解决方案**：
- 尝试使用更小的模型（如`gpt2-xl`而不是`gpt-neox-20b`）
- 关闭其他占用GPU的程序

### 问题2：下载数据集失败
**解决方案**：
- 使用浏览器直接下载：https://rome.baulab.info/data/dsets/counterfact.json
- 将下载的文件放到项目根目录

### 问题3：模型下载速度慢
**解决方案**：
- 模型会从HuggingFace自动下载，首次运行会比较慢（几GB到几十GB）
- 耐心等待，后续运行会使用缓存的模型

### 问题4：想要切换不同的模型
**解决方案**：
```bash
# 使用小模型测试（约6GB显存）
python icl.py --model_name gpt2-xl

# 使用中等模型（约16GB显存）
python icl.py --model_name EleutherAI/gpt-neo-1.3B

# 使用大模型（约24GB显存）
python icl.py --model_name EleutherAI/gpt-j-6B
```

## 📁 项目文件说明

- `icl.py` - IKE主实验代码
- `prompt.py` - PROMPT基线实验代码（需要单独提供）
- `encode_facts.py` - 为事实生成语义编码
- `semantic_search.py` - 语义相似度搜索
- `clean_paraphrase.py` - 数据清理脚本
- `corpus_idx.txt` - 预先计算的相似样本索引
- `counterfact.json` - CounterFact数据集（需要下载）
- `relations.jsonl` - 关系定义文件
- `time_editing.json` - 时间编辑相关数据
- `test_api_connection.py` - API连接测试脚本，用于测试Gemini和OpenAI API是否可以正常访问

## 💡 项目改进建议

1. **添加requirements.txt**：方便一键安装所有依赖
2. **添加更详细的错误处理**：让用户更容易发现问题
3. **添加进度保存功能**：支持中断后继续运行
4. **优化显存使用**：支持在更小的GPU上运行

## 📝 更新日志

- 2025-10-28：创建详细的中文运行指南
- 2025-10-28：修复代码，添加 `--model_name` 参数支持，现在可以通过命令行切换不同模型
- 2025-01-XX：添加API测试脚本 `test_api_connection.py`，支持测试Gemini和OpenAI API连接
