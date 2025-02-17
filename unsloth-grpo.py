from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
import torch
max_seq_length = 512*3 # Can increase for longer reasoning traces
lora_rank = 8 # Larger rank = smarter, but slower
#pip install diffusers
#!pip install "unsloth==2025.2.4" vllm
#!pip install --upgrade pillow
# Temporarily install a specific TRL nightly version
#!pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./models/deepseek-distill-qwen-1.5b-math",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        #"q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",
        "q_proj", "k_proj", "v_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
def extract_answer(text: str) -> str:
    # 找到</think>标签的位置
    end_tag_index = text.find('</think>')
    # 如果找到了</think>标签，返回该标签之后的所有文本
    if end_tag_index != -1:
        return text[end_tag_index + len('</think>'):].strip()
    else:
        # 如果没有找到</think>标签，返回空字符串或者原始文本
        return ""
from typing import List, Dict
def dataset_process(name: str = dataset_dir) -> List[Dict]:
    data = pd.read_csv(name)
    # 使用 apply 方法来逐行处理 DataFrame
    processed_data = data.apply(lambda row: {
        'prompt': [
            {'role': 'system', 'content': SYS},
            {'role': 'user', 'content': row['prompt1']}
        ],
        'answer': row['answer']
    }, axis=1).tolist()  # 将结果转换为列表
    return processed_data
# Reward functions
#COUNT=0
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>.*"  # 正则表达式
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]  # 使用DOTALL模式匹配换行符
    return [0.5 if match else 0.0 for match in matches]
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 256*5,
    max_completion_length = 512*3,
    num_train_epochs = 1, # Set to 1 for a full training run
    #max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "./result/deepseek-distill-qwen-1.5b-math-grpo",
)
import re
from datasets import load_dataset, Dataset
SYS = """You are DeepQuery, a data science expert. 
Below, you are presented with a database schema and a question.
This schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. Your task is to read the schema, understand the question, and generate a valid SQL query to answer the question. 
You should include your reasoning between <think> and </think> and then provide a SQL answer.
"""
dataset_dir = "./cot-qa-distill.csv"
import pandas as pd
def dataset_process(name: str = dataset_dir) -> List[Dict]:
    data = pd.read_csv(name)
    # 使用 apply 方法来逐行处理 DataFrame
    processed_data = data.apply(lambda row: {
        'prompt': [
            {'role': 'system', 'content': SYS},
            {'role': 'user', 'content': row['query']}
        ],
        'answer': row['answer']
    }, axis=1).tolist()  # 将结果转换为列表
    return processed_data
    
dataset = dataset_process()

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()