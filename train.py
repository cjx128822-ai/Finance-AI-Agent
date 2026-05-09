import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import swanlab
from swanlab.integration.transformers import SwanLabCallback
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_id = "qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 针对 Qwen2.5 的核心层
    lora_dropout=0.05, 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # 打印你会发现，可训练参数只占 1% 左右
def process_func(example):
    MAX_LENGTH = 384 
    input_ids, labels = [], []
    # 按照 Qwen2.5 的聊天模板拼接
    instruction = tokenizer(f"<|im_start|>system\n你是一个金融分析师<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"]
    # 标签部分：只有回答部分计算 Loss，指令部分用 -100 忽略
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    
    return {"input_ids": input_ids[:MAX_LENGTH], "labels": labels[:MAX_LENGTH]}

# 加载之前 prepare_data.py 生成的数据
dataset = load_dataset("json", data_files="data/finance_data.jsonl", split="train")
tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)

# 5. 训练参数（专为 6G 显存优化）
args = TrainingArguments(
    output_dir="./output/Qwen2.5_finance",
    per_device_train_batch_size=1, # 显存小，batch 设为 1
    gradient_accumulation_steps=8, # 累计 8 步更新一次，等效 batch=8，保证训练平稳
    logging_steps=5,
    num_train_epochs=1,            # 先跑 1 个轮次看看效果
    save_steps=50,
    learning_rate=1e-4,
    gradient_checkpointing=True,   # 开启后能大幅降低显存占用
    fp16=True,                     # 半精度加速
    report_to="none"               # 暂时关掉默认报告
)

# 6. 启动训练（加入 SwanLab 监控曲线）
swanlab_callback = SwanLabCallback(project="Qwen2.5-Finance-SFT")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]
)

print("🚀 正在起飞！开始微调...")
trainer.train()