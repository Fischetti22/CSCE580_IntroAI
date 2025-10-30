#!/usr/bin/env python3
"""
Complete the analysis using the already fine-tuned model
This saves time by not re-training
"""

import pandas as pd
import numpy as np

# Fix matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import time
import os

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    GPT2Tokenizer, GPT2ForSequenceClassification
)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

# Configuration
BATCH_SIZE = 4
MAX_LENGTH = 256
DATA_PATH = 'data/IMDB Dataset.csv'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def get_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)


print("\n" + "="*80)
print("COMPLETING ANALYSIS WITH PRE-TRAINED MODEL")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df['label'] = (df['sentiment'] == 'positive').astype(int)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'].values, df['label'].values, test_size=0.2, 
    stratify=df['label'].values, random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, 
    stratify=train_labels, random_state=42
)

print(f"Test set: {len(test_texts)} samples")

# Initialize tokenizers
print("\nInitializing tokenizers...")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Load fine-tuned model
print("\nLoading fine-tuned DistilBERT model...")
finetuned_distilbert = DistilBertForSequenceClassification.from_pretrained(
    f'{MODELS_DIR}/finetuned_distilbert'
).to(device)

# Load base models
print("Loading base DistilBERT...")
base_distilbert = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
).to(device)

print("Loading base GPT-2...")
base_gpt2 = GPT2ForSequenceClassification.from_pretrained(
    'gpt2', num_labels=2
).to(device)
base_gpt2.config.pad_token_id = gpt2_tokenizer.eos_token_id

# Train classical model
print("\nTraining Logistic Regression...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
test_tfidf = tfidf_vectorizer.transform(test_texts)

lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
lr_model.fit(train_tfidf, train_labels)
print("Logistic Regression trained")

# Prepare test data
print("\nPreparing test dataloaders...")
test_dataset = IMDBDataset(test_texts, test_labels, distilbert_tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

test_dataset_gpt2 = IMDBDataset(test_texts, test_labels, gpt2_tokenizer, MAX_LENGTH)
test_loader_gpt2 = DataLoader(test_dataset_gpt2, batch_size=BATCH_SIZE)

# Evaluate all models
print("\nEvaluating models...")
print("  1. Fine-tuned DistilBERT...")
preds_finetuned, labels_true = get_predictions(finetuned_distilbert, test_loader, device)

print("  2. Base DistilBERT...")
preds_base_distilbert, _ = get_predictions(base_distilbert, test_loader, device)

print("  3. Base GPT-2...")
preds_gpt2, _ = get_predictions(base_gpt2, test_loader_gpt2, device)

print("  4. Logistic Regression...")
preds_lr = lr_model.predict(test_tfidf)

# Calculate metrics
print("\nCalculating metrics...")
models_data = [
    ('Fine-tuned DistilBERT', preds_finetuned),
    ('Base DistilBERT', preds_base_distilbert),
    ('Base GPT-2', preds_gpt2),
    ('Logistic Regression', preds_lr)
]

metrics_results = {}
for model_name, preds in models_data:
    accuracy = accuracy_score(labels_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_true, preds, average='binary'
    )
    
    metrics_results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

metrics_df = pd.DataFrame(metrics_results).T

print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)
print(metrics_df.to_string())
print("="*70)

# Save to CSV
os.makedirs(RESULTS_DIR, exist_ok=True)
comparison_df = metrics_df.reset_index()
comparison_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
comparison_df = comparison_df.round(4)
comparison_df['Rank'] = comparison_df['F1-Score'].rank(ascending=False).astype(int)
comparison_df = comparison_df.sort_values('Rank')
comparison_df.to_csv(f'{RESULTS_DIR}/performance_comparison.csv', index=False)
print(f"\n✓ Saved: {RESULTS_DIR}/performance_comparison.csv")

# Generate confusion matrices
print("\nGenerating confusion matrices...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (model_name, preds) in enumerate(models_data):
    cm = confusion_matrix(labels_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    acc = accuracy_score(labels_true, preds)
    axes[idx].set_title(f'{model_name}\nAccuracy: {acc:.4f}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_DIR}/confusion_matrices.png")
plt.close()

# Generate metrics comparison
print("Generating metrics comparison chart...")
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(metrics_df.index))
width = 0.2

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, metrics_df[metric], width, label=metric, color=colors[i])

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Performance Metrics Comparison Across Models')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_df.index, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_DIR}/metrics_comparison.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\n🏆 Model Rankings (by F1-Score):")
for _, row in comparison_df.iterrows():
    print(f"  {row['Rank']}. {row['Model']}: {row['F1-Score']:.4f}")

print("\n📁 Generated Files:")
print(f"  - {RESULTS_DIR}/confusion_matrices.png")
print(f"  - {RESULTS_DIR}/metrics_comparison.png")
print(f"  - {RESULTS_DIR}/performance_comparison.csv")
print(f"  - {MODELS_DIR}/finetuned_distilbert/ (already existed)")

print("\n✅ All done! Check the results/ directory for outputs.")
print("="*80)
