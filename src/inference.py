# src/inference.py
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from peft import PeftModel
import torch

def load_model(model_path):
    # 检查是否是LoRA适配器
    if "lora" in model_path.lower():
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # 合并LoRA权重
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = load_model(args.model_path)
    
    # 创建pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # 执行推理
    result = classifier(args.text)
    print(f"输入文本: {args.text}")
    print(f"预测结果: {result[0]['label']} (置信度: {result[0]['score']:.2f})")

if __name__ == "__main__":
    main()
