# DeepQuery-1.5b: NL2SQL with deep thinking
# Training Scripts

This repository contains two training scripts for fine-tuning language models on Natural Language to SQL (NL2SQL) tasks using the Unsloth library. The scripts implement different training approaches: Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

## Overview

The scripts are designed to fine-tune the Qwen2.5-coder-1.5b-instruct model for converting natural language queries into SQL statements. The training process includes:

1. **SFT (Supervised Fine-Tuning)**: Initial training using labeled data with query-response pairs
2. **GRPO (Group Relative Policy Optimization)**: Reinforcement learning approach to further improve model performance

## Scripts Description

### 1. unsloth-sft.py

This script implements Supervised Fine-Tuning (SFT) using the Unsloth library.

**Key components:**
- Model initialization with Qwen-Coder-1.5b-instruct
- LoRA (Low-Rank Adaptation) configuration with rank 16
- Dataset processing for chat template formatting
- SFT training with AdamW 8-bit optimizer
- Integration with Weights & Biases for experiment tracking

```python
# Example usage
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

### 2. unsloth-grpo.py

This script implements Group Relative Policy Optimization (GRPO), a reinforcement learning approach that doesn't require separate reward and policy models.

**Key components:**
- Model initialization with DeepSeek-Distill-Qwen-1.5b-math
- Custom reward functions:
  - `correctness_reward_func`: Evaluates SQL answer correctness (0.0-2.0)
  - `strict_format_reward_func`: Checks if the completion follows the required format (0.0-0.5)
- GRPO training configuration with vLLM for fast inference
- Integration with Weights & Biases for experiment tracking

```python
# Example reward function
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
```

## Training Methodology

The training follows a two-stage approach:

1. **First Stage (SFT)**: The model is fine-tuned on a dataset containing natural language queries, SQL answers, and reasoning processes. This helps the model learn the basic task structure.

2. **Second Stage (GRPO)**: The model from the first stage undergoes reinforcement learning with GRPO. This stage focuses on:
   - Format adherence (using the `<think>...</think>` structure)
   - SQL correctness (comparing extracted answers with ground truth)

## Dataset

Both scripts use the "cot-qa" dataset, which contains:
- Natural language queries about database schemas
- Corresponding SQL answers
- Chain-of-thought reasoning processes

The dataset is available at: https://modelscope.cn/datasets/ruohuaw/sql-cot-r1-distill

## Model Resources

The trained models are available at:
- SFT model: https://modelscope.cn/models/ruohuaw/deepquery-1.5b-sft
- RL model: https://modelscope.cn/models/ruohuaw/deepquery-1.5b-rl

You can also try the demo at: https://modelscope.cn/studios/ruohuaw/deepquery-1.5b-rl


## Requirements

- Unsloth library
- PyTorch
- Transformers
- TRL (Transformer Reinforcement Learning)
- Weights & Biases (optional for tracking)

## Usage

To run the SFT training:
```bash
python unsloth-sft.py
```

To run the GRPO training:
```bash
python unsloth-grpo.py
```

## Technical Insights

### Training Experience

Our approach addressed the cold start problem by:
1. First using distilled data with complete "input-thinking process-output" examples for behavior cloning
2. Then applying reinforcement learning to encourage model exploration with only outcome rewards (not process rewards)

The GRPO algorithm was chosen because it doesn't require separate reward and policy models, significantly reducing memory usage.

### Reward Design

Our SQL parsing correctness reward provides a smoother reward signal by:
- Parsing the syntax tree of both the true SQL answer and model-generated SQL
- Evaluating components like SELECT, FROM, GROUP BY, HAVING, ORDER BY, WHERE
- Allowing for variations in column order and table join order
- Providing a continuous score (0-1) rather than binary correct/incorrect evaluation

This reward shaping helps the model gradually improve on difficult NL2SQL tasks and mitigates cold start issues.

## Notes

- The GRPO approach allows the model to explore different reasoning paths while being guided by reward signals
- The training scripts include memory usage tracking to help optimize for different hardware configurations
- Both scripts support bfloat16 precision when available for better performance
