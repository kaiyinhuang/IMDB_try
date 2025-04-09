# BERT Sentiment Analysis Fine-tuning Project

## Project Description
Based on Hugging Face Transformers and PEFT libraries, the BERT model is used to perform sentiment analysis on IMDB movie reviews (binary classification task), supporting full parameter fine-tuning and LoRA parameter efficient fine-tuning.

## Quick Start

### Installation dependencies
```bash
pip install -r requirements.txt
```

### Training the Model
#### Fine-tuning all parameters
```bash
python src/train.py \
    --model_name bert-base-uncased \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --epochs 3
```

#### LoRA Fine-tuning
```bash
python src/train_lora.py \
    --model_name bert-base-uncased \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 3 \
    --lora_r 8 \
    --lora_alpha 32
```

### Using model inference
```bash
python src/inference.py \
    --text "This movie was fantastic!" \
    --model_path ./outputs/lora_model
```
