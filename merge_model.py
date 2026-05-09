import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_model_path = "qwen/Qwen2.5-1.5B-Instruct"
lora_path = "./output/Qwen2.5_finance/checkpoint-125"
output_path = "./Qwen2.5_finance_merged"

print("🔄 1. 正在加载基座模型 (加载到 CPU 以确保融合精度)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print(f"🧩 2. 正在加载 LoRA 补丁: {lora_path} ...")
model = PeftModel.from_pretrained(base_model, lora_path)

print("⚒️ 3. 正在将 LoRA 权重物理熔铸进基座模型...")
model = model.merge_and_unload()

print(f"💾 4. 正在保存完整的金融大模型至: {output_path} ...")
if not os.path.exists(output_path):
    os.makedirs(output_path)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("✅ 大功告成！")