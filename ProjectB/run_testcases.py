#!/usr/bin/env python3
"""
Run AI Test Cases through all 4 models
Generates actual predictions and confidence scores
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    GPT2Tokenizer, GPT2ForSequenceClassification
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Test cases
test_cases = [
    {
        "id": 1,
        "name": "Simple Short Positive Review",
        "input": "This movie was absolutely wonderful! Great acting and amazing story. Highly recommend!",
        "expected": "positive",
        "complexity": "easy"
    },
    {
        "id": 2,
        "name": "Complex Negative Review with Sarcasm",
        "input": "Well, I have to say this movie certainly exceeded my expectations - I expected it to be mediocre, but it managed to be even worse. The director apparently thought that adding endless slow-motion sequences would somehow compensate for the complete lack of plot development. The lead actor delivered his lines with all the emotional depth of a cardboard cutout. Sure, the cinematography was decent, but that's like saying a car looks nice even though the engine doesn't work. I'd rather watch paint dry for two hours. Save your money and your time.",
        "expected": "negative",
        "complexity": "hard"
    },
    {
        "id": 3,
        "name": "Medium-Length Positive Review with Caveats",
        "input": "I went into this movie with low expectations, but I was pleasantly surprised. While the pacing dragged a bit in the middle and some supporting characters felt underdeveloped, the main storyline was engaging and the lead performances were genuinely moving. The twist at the end caught me completely off guard. Yes, it has flaws, but overall it's a solid film that kept me entertained. Worth watching if you enjoy character-driven dramas.",
        "expected": "positive",
        "complexity": "medium"
    }
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

def load_models():
    """Load all models"""
    print("Loading models...")
    
    # Fine-tuned DistilBERT
    print("  Loading fine-tuned DistilBERT...")
    ft_model = DistilBertForSequenceClassification.from_pretrained('models/finetuned_distilbert').to(device)
    ft_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Base DistilBERT
    print("  Loading base DistilBERT...")
    base_distilbert = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    ).to(device)
    
    # Base GPT-2
    print("  Loading base GPT-2...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    base_gpt2 = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2).to(device)
    base_gpt2.config.pad_token_id = gpt2_tokenizer.eos_token_id
    
    # Logistic Regression
    print("  Loading Logistic Regression and TF-IDF...")
    # Load the trained model (assuming you saved it during training)
    # If not saved, you'll need to retrain it
    try:
        with open('models/lr_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
    except FileNotFoundError:
        print("  Warning: LR model not found, will need to retrain...")
        # You'll need to add code here to retrain if models don't exist
        # For now, returning None
        lr_model = None
        tfidf_vectorizer = None
    
    print("All models loaded!\n")
    
    return {
        'ft_model': ft_model,
        'ft_tokenizer': ft_tokenizer,
        'base_distilbert': base_distilbert,
        'base_gpt2': base_gpt2,
        'gpt2_tokenizer': gpt2_tokenizer,
        'lr_model': lr_model,
        'tfidf_vectorizer': tfidf_vectorizer
    }

def predict_transformer(text, model, tokenizer, max_length=512):
    """Get prediction and confidence from transformer model"""
    model.eval()
    
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item() * 100
    
    sentiment = "positive" if pred_class == 1 else "negative"
    return sentiment, confidence

def predict_lr(text, lr_model, tfidf_vectorizer):
    """Get prediction and confidence from Logistic Regression"""
    if lr_model is None or tfidf_vectorizer is None:
        return "N/A", 0.0
    
    text_tfidf = tfidf_vectorizer.transform([text])
    pred = lr_model.predict(text_tfidf)[0]
    proba = lr_model.predict_proba(text_tfidf)[0]
    
    sentiment = "positive" if pred == 1 else "negative"
    confidence = proba[pred] * 100
    
    return sentiment, confidence

def run_test_cases(models):
    """Run all test cases through all models"""
    print("="*80)
    print("RUNNING TEST CASES")
    print("="*80)
    
    results = []
    
    for tc in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST CASE {tc['id']}: {tc['name']}")
        print(f"Complexity: {tc['complexity'].upper()}")
        print(f"Expected: {tc['expected'].upper()}")
        print(f"{'='*80}")
        print(f"\nInput: {tc['input'][:100]}...")
        print()
        
        # Fine-tuned DistilBERT
        print("1. Fine-tuned DistilBERT...")
        ft_sent, ft_conf = predict_transformer(
            tc['input'], models['ft_model'], models['ft_tokenizer']
        )
        print(f"   Prediction: {ft_sent.upper()} ({ft_conf:.1f}% confidence)")
        
        # Base DistilBERT
        print("2. Base DistilBERT...")
        base_distilbert_sent, base_distilbert_conf = predict_transformer(
            tc['input'], models['base_distilbert'], models['ft_tokenizer']
        )
        print(f"   Prediction: {base_distilbert_sent.upper()} ({base_distilbert_conf:.1f}% confidence)")
        
        # Base GPT-2
        print("3. Base GPT-2...")
        gpt2_sent, gpt2_conf = predict_transformer(
            tc['input'], models['base_gpt2'], models['gpt2_tokenizer']
        )
        print(f"   Prediction: {gpt2_sent.upper()} ({gpt2_conf:.1f}% confidence)")
        
        # Logistic Regression
        print("4. Logistic Regression (baseline)...")
        lr_sent, lr_conf = predict_lr(
            tc['input'], models['lr_model'], models['tfidf_vectorizer']
        )
        print(f"   Prediction: {lr_sent.upper()} ({lr_conf:.1f}% confidence)")
        
        # Store results
        results.append({
            'test_case': tc['id'],
            'name': tc['name'],
            'complexity': tc['complexity'],
            'expected': tc['expected'],
            'ft_pred': ft_sent,
            'ft_conf': ft_conf,
            'ft_correct': ft_sent == tc['expected'],
            'base_distilbert_pred': base_distilbert_sent,
            'base_distilbert_conf': base_distilbert_conf,
            'base_distilbert_correct': base_distilbert_sent == tc['expected'],
            'gpt2_pred': gpt2_sent,
            'gpt2_conf': gpt2_conf,
            'gpt2_correct': gpt2_sent == tc['expected'],
            'lr_pred': lr_sent,
            'lr_conf': lr_conf,
            'lr_correct': lr_sent == tc['expected']
        })
    
    return results

def save_results(results):
    """Save results to CSV"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/testcase_results.csv', index=False)
    print("Results saved to: results/testcase_results.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    models_to_check = [
        ('Fine-tuned DistilBERT', 'ft'),
        ('Base DistilBERT', 'base_distilbert'),
        ('Base GPT-2', 'gpt2'),
        ('Logistic Regression', 'lr')
    ]
    
    print("\nModel Performance Summary:")
    for model_name, prefix in models_to_check:
        correct_count = sum(df[f'{prefix}_correct'])
        success_rate = (correct_count / len(df)) * 100
        avg_conf = df[f'{prefix}_conf'].mean()
        print(f"  {model_name:30s} {correct_count}/3 ({success_rate:.0f}%) | Avg Confidence: {avg_conf:.1f}%")

if __name__ == "__main__":
    models = load_models()
    results = run_test_cases(models)
    save_results(results)
    print("\n✅ Test cases complete!")
