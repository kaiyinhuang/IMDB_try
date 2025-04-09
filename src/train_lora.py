# src/train_lora.py
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import evaluate

def main(config):
    # 1. 加载数据集（同train.py）
    # ...（与train.py相同的数据加载代码）...
    
    # 2. 加载基础模型
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["model_name"],
        num_labels=config["model"]["num_labels"]
    )
    
    # 3. 添加LoRA适配器
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias="none",
        modules_to_save=["classifier"]  # 分类层保持可训练
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 训练参数（学习率更大）
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        # ...（其他参数同train.py）...
    )
    
    # 5. 训练和保存（同train.py）
    # ...（后续代码与train.py相同）...

if __name__ == "__main__":
    import yaml
    with open("configs/lora.yaml") as f:
        config = yaml.safe_load(f)
    main(config)
