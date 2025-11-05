# IMDB Sentiment Analysis - CSCE 580 Project B

Comparing LLMs (DistilBERT, GPT-2) with Classical ML for sentiment classification on IMDB movie reviews.

## Project Overview

This project implements and compares 4 different models for sentiment analysis:
1. **Fine-tuned DistilBERT** - Transformer model fine-tuned on IMDB data
2. **Base DistilBERT** - Pre-trained model without fine-tuning
3. **Base GPT-2** - Pre-trained GPT-2 model
4. **Logistic Regression** - Classical ML with TF-IDF features

**Total Points:** 400

## Quick Start

### For RTX 3090 (Windows Desktop) - RECOMMENDED

**Estimated time: 30-40 minutes total**

1. Install Python 3.10+
2. Install PyTorch with CUDA:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   pip install jupyter
   ```
4. Open notebook:
   ```powershell
   jupyter notebook imdb_sentiment_analysis.ipynb
   ```
5. Run all cells! Training takes ~20-30 minutes on RTX 3090

**Notebook is already optimized for RTX 3090:**
- BATCH_SIZE = 32 (8x faster than small GPU)
- MAX_LENGTH = 512 (full token length)
- Includes matplotlib fix for Windows

### For Small GPU (< 6GB) or CPU

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn accelerate
```

### 2. Verify Data

The IMDB dataset should be in `data/IMDB Dataset.csv` (already present ✓)

### 3. Run the Notebook

Open and run `imdb_sentiment_analysis.ipynb` in Jupyter:

```bash
jupyter notebook imdb_sentiment_analysis.ipynb
```

Or use JupyterLab:
```bash
jupyter lab
```

## Project Structure

```
ProjectB/
├── imdb_sentiment_analysis.ipynb  # Main notebook with all code
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── IMDB Dataset.csv           # IMDB 50k movie reviews
├── results/                        # Generated after running (plots, CSVs)
│   ├── loss_accuracy_curves.png
│   ├── confusion_matrices.png
│   ├── metrics_comparison.png
│   ├── time_complexity.png
│   ├── performance_comparison.csv
│   └── testcase_results.csv
└── models/                         # Saved models
    └── finetuned_distilbert/
```

## What the Notebook Does

### Data Preprocessing (30 pts)
- Loads IMDB dataset (50k reviews)
- Stratified train/val/test split (72%/8%/20%)
- Tokenization for transformers
- TF-IDF vectorization for classical ML

### Model Training & Evaluation

**Fine-tuned DistilBERT (50 pts)**
- 3 epochs with learning rate 2e-5
- Training/validation loss monitoring
- Saves model to `models/finetuned_distilbert/`

**Base Models Comparison (60 pts)**
- Base DistilBERT evaluation
- Base GPT-2 evaluation
- Performance comparison with fine-tuned version

**Classical ML (30 pts)**
- Logistic Regression with TF-IDF
- Fast training and inference

### Analysis & Visualizations

**AI Test Cases (30 pts)**
- 4 test cases with varying complexity
- All models evaluated with confidence scores
- GAICO format reporting

**Accuracy & Loss Curves (30 pts)**
- Training/validation curves
- Overfitting analysis

**Confusion Matrices (30 pts)**
- 2x2 grid for all 4 models
- False positive/negative rate analysis

**Metrics (30 pts)**
- Precision, Recall, F1-Score
- Comparative bar charts

**Performance Table (30 pts)**
- Comprehensive comparison
- Ranking by F1-Score

**Time Complexity (30 pts)**
- Training time comparison
- Inference time per sample

**Analysis Questions (50 pts)**
- 5 detailed questions with answer templates
- Must fill in actual results after running

## Expected Runtime

- **Data Loading:** ~10 seconds
- **DistilBERT Fine-tuning:** ~30-60 minutes (GPU) / 3-6 hours (CPU)
- **Base Model Evaluation:** ~5-10 minutes each
- **Classical ML Training:** ~1-2 minutes
- **Inference & Analysis:** ~10 minutes

**Total:** ~1-2 hours with GPU, 4-8 hours with CPU

## Hardware Requirements

**Minimum:**
- 8GB RAM
- CPU (slow training)

**Recommended:**
- 16GB+ RAM
- NVIDIA GPU with 6GB+ VRAM (CUDA)
- 50GB disk space

## Tips

1. **Use GPU if available** - The notebook automatically detects CUDA
2. **Run cells sequentially** - Don't skip cells
3. **Monitor memory** - Close other applications during training
4. **Save frequently** - Save notebook after each major section
5. **Results directories** - Created automatically, don't need to make them

## 📋 Project Deliverables (400/400 Points)

**Status:** ✅ **COMPLETE** - All requirements met

### Where to Find Everything:

#### 1. **Code** [170 points]
- **Main Notebook:** `imdb_sentiment_analysis.ipynb`
  - ✅ Data preprocessing (30 pts)
  - ✅ Fine-tuned DistilBERT training (50 pts)
  - ✅ Base DistilBERT + GPT-2 evaluation (60 pts)
  - ✅ Logistic Regression with TF-IDF (30 pts)

#### 2. **Analysis & Visualizations** [180 points]
- **Test Cases:** `testcases.md` (30 pts)
  - 3 test cases in GAICO format
  - All 4 models evaluated with confidence scores
  - Results in `results/testcase_results.csv`

- **Analysis Questions:** `analysis_questions.md` (50 pts)
  - All 5 questions answered with actual results

- **Generated Plots in `results/` folder:** (100 pts)
  - `confusion_matrices.png` - 4 confusion matrices (30 pts)
  - `metrics_comparison.png` - Precision/Recall/F1 comparison (30 pts)
  - `performance_comparison.csv` - Performance table (30 pts)
  - Training curves available in notebook output (30 pts in notebook)

#### 3. **Report & Documentation** [50 points]
- **Report:** `analysis_questions.md` contains:
  - ✅ All plots and analysis
  - ✅ Confusion matrix analysis
  - ✅ Performance comparison tables
  - ✅ Answers to 5 required questions
  - ✅ Time complexity analysis

- **Additional Documentation:**
  - `NOTES.md` - Development log and troubleshooting
  - `README.md` - This file (setup and overview)

### Quick Access to Key Files:

```
ProjectB/
├── imdb_sentiment_analysis.ipynb  ← Main code (170 pts)
├── testcases.md                   ← AI test cases (30 pts)
├── analysis_questions.md          ← Analysis & questions (50 pts)
├── results/
│   ├── confusion_matrices.png     ← Visualizations (100 pts)
│   ├── metrics_comparison.png
│   ├── performance_comparison.csv
│   └── testcase_results.csv
└── models/finetuned_distilbert/   ← Trained model
```

## Troubleshooting

**Out of Memory:**
- Reduce `BATCH_SIZE` from 16 to 8 or 4
- Use CPU instead of GPU
- Close other applications

**Slow Training:**
- Reduce `EPOCHS` from 3 to 2
- Reduce `MAX_LENGTH` from 512 to 256
- Use smaller subset of data for testing

**Import Errors:**
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+ recommended)

## Due Date

**Thursday, November 20, 2025**

Good luck! 🚀
