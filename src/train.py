# src/train.py
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

def main(config):
    # 1. 加载数据集
    dataset = load_dataset(config["data"]["dataset_name"])
    
    # 2. 数据预处理
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config["data"]["max_length"]
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    
    # 3. 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["model_name"],
        num_labels=config["model"]["num_labels"]
    )
    
    # 4. 定义评估指标
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    # 5. 训练参数设置
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=config["training"]["fp16"],
        report_to="wandb" if config["training"]["use_wandb"] else "none",
    )
    
    # 6. 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 7. 启动训练
    trainer.train()
    
    # 8. 保存模型
    model.save_pretrained(config["training"]["output_dir"])

if __name__ == "__main__":
    import yaml
    # 加载配置文件
    with open("configs/full_ft.yaml") as f:
        config = yaml.safe_load(f)
    main(config)
