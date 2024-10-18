import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# 检查是否有 CUDA 可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. 加载预训练的 LLaMA 模型和 tokenizer
model_name="cognitivecomputations/dolphin-2.9.3-qwen2-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="your hf token")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

#使用本地模型和配置
#model = AutoModelForCausalLM.from_pretrained("./pretrained", local_files_only=True, load_in_8bit=True).to(device)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    bias="none", 
    task_type="dolphin"
)

# 3. 将模型转换为可微调的 LoRA 模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 检查微调的参数

# 4. 准备训练数据集
# 这里我们使用 Hugging Face 的 datasets 库加载一个简单的样本数据集
dataset = load_dataset("text", data_files={"train": "./data.txt","test": "./test.txt"})

# 数据预处理：tokenizer 处理输入文本
def preprocess_function(examples):
    tokenized = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  # 将 input_ids 作为 labels
    return tokenized

train_dataset = dataset["train"].map(preprocess_function, batched=True)
test_dataset = dataset["test"].map(preprocess_function, batched=True)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./lora-finetune-llama",
    per_device_train_batch_size=4,   # 根据显存大小调整 batch size
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,  # 根据显存大小调整
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    report_to="none",  # 禁用 W&B 或其他报告
    fp16=True,         # 启用混合精度训练（如果使用的是 CUDA）
    load_best_model_at_end=True
)

# 6. 使用 Trainer 进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 7. 开始训练
trainer.train()

# 8. 保存模型
trainer.save_model("./lora-finetuned-llama")

