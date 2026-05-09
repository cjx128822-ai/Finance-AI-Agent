import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 注意：这里改成你目录里数字最大的那个
lora_path = "./output/Qwen2.5_finance/checkpoint-125"
base_model_path = "qwen/Qwen2.5-1.5B-Instruct"

print("正在加载模型并合体 LoRA 权重...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# 用半精度加载，省显存
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 关键：把 LoRA 补丁贴到原模型上
model = PeftModel.from_pretrained(model, model_id=lora_path)

def chat(text):
    inputs = tokenizer(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print("-" * 30)
print("问：什么是市盈率？")
print(f"答：{chat('什么是市盈率？')}")