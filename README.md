# BERT Text Classification Fine-tuning Project

## Project Description
Based on the Hugging Face Transformers library, use the BERT model to perform sentiment analysis (binary classification) on IMDB movie reviews.

## Quick Start
1. Installation dependencies:
   ```bash
   pip install -r requirements.txt

2. Run training:
   ```bash
   python src/train.py
   --model_name bert-base-uncased \
    --lr 2e-5 \
    --batch_size 8 \
    --epochs 3

4. Model inference:
   ```bash
   python src/inference.py
   --text "This movie was amazing!"
   --model_dir ./saved_model
