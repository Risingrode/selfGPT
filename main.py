import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import get_linear_schedule_with_warmup

# 1. 加载和解析数据
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 转换数据格式
formatted_data = []
for item in data:
    formatted_data.append({
        "instruction": item["问题"],
        "input": item["条款"],
        "output": item["答案"]
    })

# 3. 创建数据集
dataset = Dataset.from_list(formatted_data)

# 4. 加载预训练的 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 替换为你选择的更强大的模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 5. 定义 tokenize 函数
def tokenize_function(examples):
    return tokenizer(examples['instruction'], examples['input'], examples['output'],
                     truncation=True, padding="max_length", max_length=512)

# 6. 应用 tokenize 函数到数据集
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 7. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # 根据计算资源调整
    per_device_eval_batch_size=8,
    num_train_epochs=40,  # 增加训练轮次
    weight_decay=0.01,
    save_total_limit=2,  # 只保留最新的模型
    save_steps=1000,
    fp16=True,  # 启用混合精度训练
    gradient_accumulation_steps=4,  # 梯度累积，以模拟更大的 batch size
    logging_dir='./logs',
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# 8. 使用学习率调度器和 warmup
total_steps = len(tokenized_dataset) * training_args.num_train_epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    training_args.optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 9. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    optimizers=(training_args.optimizer, scheduler)
)

# 10. 开始训练
trainer.train()

# 11. 保存最佳模型
trainer.save_model("best_model")
