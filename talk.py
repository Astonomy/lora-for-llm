import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file as load_safetensors
from peft import PeftModel, LoraConfig, get_peft_model

# 定义模型和 LoRA 权重的路径
model_name = "cognitivecomputations/dolphin-2.9.3-qwen2-0.5b"  # 例如 GPT-2 模型架构
safetensor_lora_path = "./lora-finetune-llama/checkpoint-2361/adapter_model.safetensors"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型架构
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置 LoRA 配置
lora_config = LoraConfig(
    r=8,                      # LoRA rank
    lora_alpha=32,             # LoRA scaling factor
    lora_dropout=0.1,          # Dropout rate
    bias="none",               # 无 bias 适应
    task_type="CAUSAL_LM"      # 用于因果语言建模任务
)

# 将模型转换为支持 LoRA 的模型
model = get_peft_model(model, lora_config)

# 加载 LoRA 权重（也是 safetensors 格式）
lora_state_dict = load_safetensors(safetensor_lora_path)
model.load_state_dict(lora_state_dict, strict=False)

# 将模型切换到评估模式
model.eval()

# 使用 GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using " + device)
model.to(device)

# 准备推理输入
input_text = input("you: ")
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 添加生成参数
max_new_tokens = 200  # 生成最大新 token 数
temperature = 0.9     # 控制生成文本的多样性
top_p = 0.9           # nucleus sampling 策略的 top-p
print("Generating text...")

# 推理
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,  # 使用随机采样
        pad_token_id=tokenizer.eos_token_id  # 防止错误
    )

# 输出生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

