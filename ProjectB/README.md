# IMDB Sentiment Analysis - CSCE 580 Project B
 
**Due Date:** November 20, 2025  
**Completed:** November 5, 2025

This project compares LLMs (DistilBERT, GPT-2) with classical machine learning for sentiment classification on IMDB movie reviews.

---

## Project Structure

```
ProjectB/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── CSCE580-Fall2025-Project_...pdf        # Project requirements
│
├── notebooks/                             # Jupyter Notebooks
│   └── imdb_sentiment_analysis.ipynb     # Main notebook
│
├── src/                                   # Python Scripts
│   ├── imdb_analysis.py                  # Main training script
│   ├── complete_analysis.py              # Quick analysis (if model exists)
│   └── run_testcases.py                  # Test case evaluation
│
├── data/                                  # Dataset
│   └── IMDB Dataset.csv                  # IMDB 50k movie reviews
│
├── models/                                # Trained Models
│   └── finetuned_distilbert/             # Fine-tuned DistilBERT
│
├── results/                               # Main Results
│   ├── confusion_matrices.png
│   ├── metrics_comparison.png
│   ├── performance_comparison.csv
│   ├── testcase_results.csv
│   └── loss_accuracy_curves.png
│
├── results_GPU3090_base/                  # GPU Results (Base config)
│   └── (same files as above)
│
├── results_GPU3090_High/                  # GPU Results (High config)
│   └── (same files as above)
│
└── docs/                                  # Documentation & Analysis
    ├── testcases.md                      # AI test cases (GAICO format)
    ├── analysis_questions.md             # Analysis & 5 questions answered
    └── NOTES.md                          # Development log
```

---

## How to Run

I've provided two ways to run this project - you can use either one:

### Option 1: Jupyter Notebook 

If you just want to run everything quickly with Jupyter:


```bash
# Install dependencies
pip install -r requirements.txt
pip install jupyter

# Launch notebook
jupyter notebook notebooks/imdb_sentiment_analysis.ipynb

# Run all cells in order
```

### Option 2: Python Script (What I Used)

I found this better for seeing results step-by-step and debugging as I run on a Linux terminal:

```bash
# Full training and analysis
python3 src/imdb_analysis.py

# If the model is already trained
python3 src/complete_analysis.py

# Just run test cases
python3 src/run_testcases.py
```

Both methods produce the same results.

---

## Hardware Requirements

I ran this on different machines and here's what I found:

**My RTX 3090 (24GB) - Best Option:**
- Training time: 30-40 minutes
- Config: BATCH_SIZE = 32, MAX_LENGTH = 512
- This is what I used for final results

**Smaller GPU (< 6GB):**
- Training time: 2-3 hours  
- Config: BATCH_SIZE = 4, MAX_LENGTH = 256
- I had to use this on my laptop initially

**CPU - Don't Recommend:**
- Training time: 30-80 hours
- Only use if you have no GPU

---

## Deliverables (400/400 Points) I hope

Here's where to find everything for grading:

| Deliverable | Location | Points |
|-------------|----------|--------|
| Main Code | `notebooks/imdb_sentiment_analysis.ipynb` | 170 pts |
| | Alternative: `src/imdb_analysis.py` | |
| Test Cases (GAICO) | `docs/testcases.md` | 30 pts |
| Analysis Questions | `docs/analysis_questions.md` | 50 pts |
| Confusion Matrices | `results/confusion_matrices.png` | 30 pts |
| Metrics Chart | `results/metrics_comparison.png` | 30 pts |
| Performance Table | `results/performance_comparison.csv` | 30 pts |
| Test Results | `results/testcase_results.csv` | 10 pts |
| Training Curves | In notebook output | 30 pts |
| Time Complexity | In notebook/docs | 30 pts |
| Development Notes | `docs/NOTES.md` | - |

### Breakdown:

**Code (170 points):**
- Data preprocessing (30 pts)
- Fine-tuned DistilBERT training (50 pts)
- Base DistilBERT + GPT-2 evaluation (60 pts)
- Logistic Regression with TF-IDF (30 pts)

**Analysis & Visualizations (180 points):**
- AI test cases in GAICO format (30 pts)
- Accuracy & loss curves (30 pts)
- Confusion matrices (30 pts)
- Precision, recall, F1-score (30 pts)
- Performance comparison table (30 pts)
- Time complexity analysis (30 pts)

**Questions (50 points):**
- All 5 analysis questions answered

---

## What I Built

I implemented and compared 4 different models for sentiment analysis:

1. **Fine-tuned DistilBERT** - Transformer model fine-tuned on IMDB data
2. **Base DistilBERT** - Pre-trained model without fine-tuning
3. **Base GPT-2** - Pre-trained GPT-2 model
4. **Logistic Regression** - Classical ML with TF-IDF features

### My Results:

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Fine-tuned DistilBERT | 93.69% | 93.73% | ~30 min (RTX 3090) |
| Logistic Regression | 89.89% | 89.95% | ~2 min |
| Base GPT-2 | 50.02% | 66.56% | N/A (pre-trained) |
| Base DistilBERT | 50.00% | 0.16% | N/A (pre-trained) |

**What surprised me:** Logistic Regression performed really well - only 3.78% lower F1-score than the fine-tuned transformer while being 15x faster. The base models without fine-tuning were essentially useless, which really shows why domain-specific training matters.

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if you want to install individually:
```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn accelerate jupyter
```

### 2. Verify Data

The IMDB dataset should be in `data/IMDB Dataset.csv` (it's already there)

### 3. Run

Choose notebook or script (see "How to Run" section above)

---

## What Gets Generated

When you run the code, it will create:

- **Models:** Saved to `models/finetuned_distilbert/`
- **Plots:** Generated in `results/` folder
- **CSVs:** Performance metrics and test results
- **Analysis:** Already complete in `docs/` folder

---

## Tips from My Experience

1. **Use a GPU** - The code automatically detects CUDA. CPU training takes forever.
2. **Run cells in order** - Don't skip around, especially in the notebook
3. **Watch your memory** - Close other applications during training
4. **Notebook vs Script** - I used the notebook for development, script for final runs

---

## Troubleshooting

Things I ran into and how I fixed them:

**Out of Memory:**
- Reduce BATCH_SIZE: 32 → 16 → 8 → 4
- Reduce MAX_LENGTH: 512 → 256
- Close other applications
- See my `docs/NOTES.md` for detailed fixes

**Slow Training:**
- Reduce EPOCHS from 3 to 2
- Use a smaller dataset for testing
- Try the script instead of notebook (slightly faster)

**Import Errors:**
- Run `pip install -r requirements.txt` again
- Check Python version (I used 3.10, need at least 3.8)

---

## Additional Documentation

- **Project Requirements:** See PDF in root directory
- **Development Log:** `docs/NOTES.md` - How I solved problems during development
- **Test Cases:** `docs/testcases.md` - GAICO format with all results
- **Analysis:** `docs/analysis_questions.md` - All questions answered with my findings

---

## Completion Status

Everything is complete and tested:

- [x] All code implemented (notebook + scripts)
- [x] All 4 models trained and evaluated
- [x] Test cases created in GAICO format
- [x] All visualizations generated
- [x] All 5 analysis questions answered with actual results
- [x] Documentation complete
- [x] Repository organized
- [x] Ready for submission

The project ran successfully on my RTX 3090 and all results are in the `results/` folder. I also have different results that show the different outputs depending on the GPU.
