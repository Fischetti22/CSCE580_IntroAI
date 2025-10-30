#!/usr/bin/env python3
"""
IMDB Sentiment Analysis - CSCE 580 Project B
Comparing Fine-tuned DistilBERT, Base DistilBERT, GPT-2, and Classical ML
"""

import pandas as pd
import numpy as np

# Fix matplotlib backend issue - use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

import time
import os
from pathlib import Path

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Transformers
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    GPT2Tokenizer, GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup
)

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Configuration
BATCH_SIZE = 4  # Reduced from 16 for low memory GPU
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients to simulate batch_size=16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 256  # Reduced from 512 to save memory
DATA_PATH = 'data/IMDB Dataset.csv'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device - Use CPU if GPU is too small (< 6GB recommended)
if torch.cuda.is_available():
    # Clear CUDA cache
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {total_memory:.2f} GB")
    
    if total_memory < 5.0:
        print(f"\n⚠️  WARNING: GPU has only {total_memory:.2f}GB memory.")
        print("   Training will be slower but should work with reduced batch size.")
        print("   If you get OOM errors, consider using CPU by setting device='cpu' manually.")
    
    device = torch.device('cuda')
else:
    print("No GPU detected, using CPU (this will be slow)")
    device = torch.device('cpu')

print(f"Using device: {device}")


class IMDBDataset(Dataset):
    """Custom Dataset for IMDB reviews"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
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


def load_and_preprocess_data():
    """Load and preprocess IMDB dataset"""
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} reviews")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Convert labels
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    
    # Split data
    print("\nSplitting data...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['review'].values,
        df['label'].values,
        test_size=0.2,
        stratify=df['label'].values,
        random_state=42
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        stratify=train_labels,
        random_state=42
    )
    
    print(f"Training set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    return df, train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def prepare_transformer_data(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):
    """Prepare data for transformer models"""
    print("\nInitializing tokenizers...")
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    print("Creating datasets and dataloaders...")
    train_dataset = IMDBDataset(train_texts, train_labels, distilbert_tokenizer, MAX_LENGTH)
    val_dataset = IMDBDataset(val_texts, val_labels, distilbert_tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_texts, test_labels, distilbert_tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return distilbert_tokenizer, gpt2_tokenizer, train_loader, val_loader, test_loader


def prepare_classical_data(train_texts, val_texts, test_texts, train_labels):
    """Prepare TF-IDF features for classical ML"""
    print("\nCreating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    val_tfidf = tfidf_vectorizer.transform(val_texts)
    test_tfidf = tfidf_vectorizer.transform(test_texts)
    
    print(f"TF-IDF shape: {train_tfidf.shape}")
    
    return tfidf_vectorizer, train_tfidf, val_tfidf, test_tfidf


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Scale loss for gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        # Update weights every GRADIENT_ACCUMULATION_STEPS
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Clear cache periodically
        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
    
    return total_loss / len(dataloader), correct / total


def eval_model(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def finetune_distilbert(train_loader, val_loader):
    """Fine-tune DistilBERT model"""
    print("\n" + "="*80)
    print("STEP 2: FINE-TUNING DISTILBERT")
    print("="*80)
    
    print("\nInitializing model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    print(f"Training for {EPOCHS} epochs...")
    training_stats = []
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = eval_model(model, val_loader, device)
        
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    training_time = time.time() - start_time
    print(f"\nFine-tuning completed in {training_time:.2f} seconds")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_pretrained(f'{MODELS_DIR}/finetuned_distilbert')
    print(f"Model saved to {MODELS_DIR}/finetuned_distilbert")
    
    return model, training_stats, training_time


def load_base_models(gpt2_tokenizer):
    """Load base models without fine-tuning"""
    print("\n" + "="*80)
    print("STEP 3: LOADING BASE MODELS")
    print("="*80)
    
    print("\nLoading base DistilBERT...")
    base_distilbert = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    ).to(device)
    
    print("Loading base GPT-2...")
    base_gpt2 = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2).to(device)
    base_gpt2.config.pad_token_id = gpt2_tokenizer.eos_token_id
    
    print("Base models loaded successfully")
    return base_distilbert, base_gpt2


def train_classical_model(train_tfidf, val_tfidf, train_labels, val_labels):
    """Train classical ML model"""
    print("\n" + "="*80)
    print("STEP 4: TRAINING CLASSICAL ML MODEL")
    print("="*80)
    
    print("\nTraining Logistic Regression...")
    start_time = time.time()
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)  # n_jobs=1 to avoid resource leaks
    lr_model.fit(train_tfidf, train_labels)
    
    training_time = time.time() - start_time
    print(f"Logistic Regression trained in {training_time:.2f} seconds")
    
    val_preds = lr_model.predict(val_tfidf)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    return lr_model, training_time


def get_predictions(model, dataloader, device):
    """Get predictions from transformer model"""
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


def evaluate_all_models(finetuned_distilbert, base_distilbert, base_gpt2, lr_model, 
                        test_loader, test_tfidf, gpt2_tokenizer, test_texts, test_labels):
    """Evaluate all models on test set"""
    print("\n" + "="*80)
    print("STEP 5: EVALUATING ALL MODELS")
    print("="*80)
    
    # Prepare GPT-2 dataloader
    test_dataset_gpt2 = IMDBDataset(test_texts, test_labels, gpt2_tokenizer, MAX_LENGTH)
    test_loader_gpt2 = DataLoader(test_dataset_gpt2, batch_size=BATCH_SIZE)
    
    print("\n1. Fine-tuned DistilBERT")
    start_time = time.time()
    preds_finetuned, labels_true = get_predictions(finetuned_distilbert, test_loader, device)
    finetuned_inference_time = time.time() - start_time
    print(f"   Inference time: {finetuned_inference_time:.2f}s")
    
    print("\n2. Base DistilBERT")
    start_time = time.time()
    preds_base_distilbert, _ = get_predictions(base_distilbert, test_loader, device)
    base_distilbert_inference_time = time.time() - start_time
    print(f"   Inference time: {base_distilbert_inference_time:.2f}s")
    
    print("\n3. Base GPT-2")
    start_time = time.time()
    preds_gpt2, _ = get_predictions(base_gpt2, test_loader_gpt2, device)
    gpt2_inference_time = time.time() - start_time
    print(f"   Inference time: {gpt2_inference_time:.2f}s")
    
    print("\n4. Logistic Regression")
    start_time = time.time()
    preds_lr = lr_model.predict(test_tfidf)
    lr_inference_time = time.time() - start_time
    print(f"   Inference time: {lr_inference_time:.2f}s")
    
    return {
        'preds_finetuned': preds_finetuned,
        'preds_base_distilbert': preds_base_distilbert,
        'preds_gpt2': preds_gpt2,
        'preds_lr': preds_lr,
        'labels_true': labels_true,
        'inference_times': {
            'finetuned': finetuned_inference_time,
            'base_distilbert': base_distilbert_inference_time,
            'gpt2': gpt2_inference_time,
            'lr': lr_inference_time
        }
    }


def plot_training_curves(training_stats):
    """Plot accuracy and loss curves"""
    print("\n" + "="*80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\nPlotting training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = [s['epoch'] for s in training_stats]
    train_loss = [s['train_loss'] for s in training_stats]
    val_loss = [s['val_loss'] for s in training_stats]
    train_acc = [s['train_acc'] for s in training_stats]
    val_acc = [s['val_acc'] for s in training_stats]
    
    # Loss
    axes[0].plot(epochs, train_loss, 'b-o', label='Training Loss')
    axes[0].plot(epochs, val_loss, 'r-o', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, train_acc, 'b-o', label='Training Accuracy')
    axes[1].plot(epochs, val_acc, 'r-o', label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/loss_accuracy_curves.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {RESULTS_DIR}/loss_accuracy_curves.png")
    plt.close()


def plot_confusion_matrices(predictions, labels_true):
    """Plot confusion matrices for all models"""
    print("\nPlotting confusion matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    models_data = [
        ('Fine-tuned DistilBERT', predictions['preds_finetuned']),
        ('Base DistilBERT', predictions['preds_base_distilbert']),
        ('Base GPT-2', predictions['preds_gpt2']),
        ('Logistic Regression', predictions['preds_lr'])
    ]
    
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
    print(f"   Saved: {RESULTS_DIR}/confusion_matrices.png")
    plt.close()


def calculate_metrics(predictions, labels_true):
    """Calculate all metrics"""
    print("\nCalculating metrics...")
    
    models_data = [
        ('Fine-tuned DistilBERT', predictions['preds_finetuned']),
        ('Base DistilBERT', predictions['preds_base_distilbert']),
        ('Base GPT-2', predictions['preds_gpt2']),
        ('Logistic Regression', predictions['preds_lr'])
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
    comparison_df = metrics_df.reset_index()
    comparison_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    comparison_df = comparison_df.round(4)
    comparison_df['Rank'] = comparison_df['F1-Score'].rank(ascending=False).astype(int)
    comparison_df = comparison_df.sort_values('Rank')
    comparison_df.to_csv(f'{RESULTS_DIR}/performance_comparison.csv', index=False)
    print(f"\n   Saved: {RESULTS_DIR}/performance_comparison.csv")
    
    return metrics_df, comparison_df


def print_summary(training_stats, predictions, comparison_df, ft_training_time, lr_training_time):
    """Print project summary"""
    print("\n" + "="*80)
    print("PROJECT SUMMARY")
    print("="*80)
    
    print("\n🏆 Model Performance Rankings (by F1-Score):")
    for _, row in comparison_df.iterrows():
        print(f"  {row['Rank']}. {row['Model']}: {row['F1-Score']:.4f}")
    
    print("\n⚡ Training Times:")
    print(f"  Fine-tuned DistilBERT: {ft_training_time:.2f}s")
    print(f"  Logistic Regression: {lr_training_time:.2f}s")
    
    print("\n⚡ Inference Times (total):")
    for name, time_val in predictions['inference_times'].items():
        print(f"  {name}: {time_val:.2f}s")
    
    print("\n📁 Generated Files:")
    print(f"  - {RESULTS_DIR}/loss_accuracy_curves.png")
    print(f"  - {RESULTS_DIR}/confusion_matrices.png")
    print(f"  - {RESULTS_DIR}/performance_comparison.csv")
    print(f"  - {MODELS_DIR}/finetuned_distilbert/")
    
    print("\n✅ Analysis Complete!")
    print("="*80)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("IMDB SENTIMENT ANALYSIS - CSCE 580 PROJECT B")
    print("Comparing LLMs and Classical ML for Sentiment Classification")
    print("="*80)
    
    # Step 1: Load and preprocess data
    df, train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_preprocess_data()
    
    # Prepare transformer data
    distilbert_tokenizer, gpt2_tokenizer, train_loader, val_loader, test_loader = prepare_transformer_data(
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    )
    
    # Prepare classical ML data
    tfidf_vectorizer, train_tfidf, val_tfidf, test_tfidf = prepare_classical_data(
        train_texts, val_texts, test_texts, train_labels
    )
    
    # Step 2: Fine-tune DistilBERT
    finetuned_distilbert, training_stats, ft_training_time = finetune_distilbert(train_loader, val_loader)
    
    # Step 3: Load base models
    base_distilbert, base_gpt2 = load_base_models(gpt2_tokenizer)
    
    # Step 4: Train classical model
    lr_model, lr_training_time = train_classical_model(train_tfidf, val_tfidf, train_labels, val_labels)
    
    # Step 5: Evaluate all models
    predictions = evaluate_all_models(
        finetuned_distilbert, base_distilbert, base_gpt2, lr_model,
        test_loader, test_tfidf, gpt2_tokenizer, test_texts, test_labels
    )
    
    # Step 6: Generate visualizations and metrics
    plot_training_curves(training_stats)
    plot_confusion_matrices(predictions, predictions['labels_true'])
    metrics_df, comparison_df = calculate_metrics(predictions, predictions['labels_true'])
    
    # Print summary
    print_summary(training_stats, predictions, comparison_df, ft_training_time, lr_training_time)


if __name__ == "__main__":
    main()
