# inference.py
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="This movie was fantastic!")
    parser.add_argument("--model_dir", type=str, default="./saved_model")
    args = parser.parse_args()

    # 1. 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # 2. 创建推理 Pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # 自动选择 GPU/CPU
    )

    # 3. 执行推理
    result = classifier(args.text)
    print(f"输入文本: {args.text}")
    print(f"预测结果: {result[0]['label']} (置信度: {result[0]['score']:.2f})")

if __name__ == "__main__":
    main()
