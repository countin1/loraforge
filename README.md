# 🔧 LoRAForge — 大模型微调 Pipeline

> 端到端的 LoRA/QLoRA 微调 + 评测 pipeline，从数据准备到统计验证。

## ✨ 核心功能

- **数据准备**：加载、增强、分层划分
- **LoRA 微调**：支持 Qwen/LLaMA 等模型
- **QLoRA**：4bit 量化，省显存 70%
- **超参搜索**：r/alpha/lr/epochs 网格搜索
- **统计验证**：配对 t 检验 + Cohen's d 效应量

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 准备数据
python scripts/prepare_data.py --augment 3

# LoRA 微调（需要 GPU）
python scripts/train.py --model Qwen/Qwen2.5-7B --epochs 3

# QLoRA（省显存）
python scripts/train.py --model Qwen/Qwen2.5-7B --epochs 3 --qlora

# 评测对比
python scripts/evaluate.py --base-model Qwen/Qwen2.5-7B --adapter-path outputs/adapters/lora_adapter
```

## 📊 使用示例

```python
from loraforge import LoRAForge
from loraforge.data.loader import DataLoader
from loraforge.data.augment import DataAugmentor
from loraforge.data.splitter import DataSplitter

# 准备数据
loader = DataLoader("data/questions.json")
questions = loader.load()
data = loader.to_chat_format(questions)

augmentor = DataAugmentor()
augmented = augmentor.augment(data, n_variants=3)

splitter = DataSplitter()
train_data, test_data = splitter.split(augmented)

# 训练
from loraforge.train.lora import LoRATrainer
trainer = LoRATrainer("Qwen/Qwen2.5-7B")
result = trainer.train(train_data, test_data, epochs=3)

# 评测
from loraforge.eval.compare import ModelComparator
comparator = ModelComparator()
# ... 加载模型并评测
```

## 📁 项目结构

```
loraforge/
├── loraforge/
│   ├── data/
│   │   ├── loader.py       # 数据加载
│   │   ├── augment.py      # 数据增强
│   │   └── splitter.py     # 分层划分
│   ├── train/
│   │   ├── lora.py         # LoRA 训练
│   │   └── qlora.py        # QLoRA 训练
│   ├── eval/
│   │   ├── compare.py      # 前后对比
│   │   └── stats.py        # 统计检验
│   └── __init__.py
├── data/
│   └── questions.json      # 评测题目
├── scripts/
│   ├── prepare_data.py     # 数据准备
│   ├── train.py            # 训练脚本
│   └── evaluate.py         # 评测脚本
├── outputs/
│   ├── adapters/           # LoRA adapter
│   └── reports/            # 评测报告
└── README.md
```

## 🎯 面试话术

> "我用 LoRA 微调了 Qwen2.5-7B，训练数据是从经济学评测题构造的，每题增强了 3 个变体。
> 微调后在统计知识维度上 Cohen's d = 0.73，配对 t 检验 p<0.01，显著优于基座模型。
> 用 QLoRA 4bit 量化后，7B 模型只需要 6GB 显存，RTX 3060 就能跑。"

## 📚 GPU 要求

| 模型 | LoRA | QLoRA |
|------|------|-------|
| Qwen2.5-3B | 8GB | 6GB |
| Qwen2.5-7B | 16GB | 8GB |
| Qwen2.5-14B | 28GB | 12GB |
| LLaMA-3-8B | 16GB | 8GB |

## 📚 依赖

- torch >= 2.0.0
- transformers >= 4.30.0
- peft >= 0.4.0
- datasets >= 2.14.0
- accelerate >= 0.21.0
- bitsandbytes >= 0.40.0
- scipy >= 1.10.0
- numpy >= 1.24.0

## 📄 License

MIT
