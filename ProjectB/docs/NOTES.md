# Project Notes - IMDB Sentiment Analysis

## Issues I Encountered and How I Fixed Them

### Issue 1: CUDA Out of Memory (OOM)
**Date:** 2025-10-29  
**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 MiB. 
GPU 0 has a total capacity of 3.63 GiB of which 10.69 MiB is free.
```

**What Happened:** My GPU only has 3.63GB memory, and I started with:
- BATCH_SIZE = 16
- MAX_LENGTH = 512
- This needs ~6-8GB GPU memory for DistilBERT training

**How I Fixed It:**
1. Reduced `BATCH_SIZE` from 16 to 4 (4x less memory)
2. Reduced `MAX_LENGTH` from 512 to 256 (2x less memory)
3. Added gradient accumulation (accumulate 4 batches = effective batch size 16)
4. Added `torch.cuda.empty_cache()` every 50 batches
5. Added memory detection and warnings

**Files I Modified:** `imdb_analysis.py`

**Status:** ✅ Fixed

---

### Issue 2: Qt/matplotlib Display Error
**Date:** 2025-10-29  
**Error:**
```
QObject::moveToThread: Current thread is not the object's thread.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
This application failed to start because no Qt platform plugin could be initialized.
IOT instruction (core dumped)
```

**What Happened:** 
- matplotlib was trying to display plots interactively
- I'm missing or have conflicting Qt libraries for X11 display
- My system doesn't have a GUI display configured properly

**How I Fixed It:** 
1. Set matplotlib backend to 'Agg' (non-interactive) before importing pyplot
2. Added `matplotlib.use('Agg')` at the top of my imports
3. Now plots save without needing to display them

**Files I Modified:** `imdb_analysis.py`

**Status:** ✅ Fixed

---

### Issue 3: Joblib Resource Leak Warning
**Date:** 2025-10-29  
**Error:**
```
resource_tracker: There appear to be 6 leaked semlock objects to clean up at shutdown
resource_tracker: There appear to be 1 leaked folder objects to clean up at shutdown
```

**What Happened:** Multiprocessing resources from scikit-learn weren't getting cleaned up properly

**How I Fixed It:** 
1. Changed `n_jobs=-1` to `n_jobs=1` in LogisticRegression
2. Training is slightly slower but no more resource leaks
3. No impact on final model quality

**Files I Modified:** `imdb_analysis.py`

**Status:** ✅ Fixed

---

## Configuration Changes Log

### Original Configuration
```python
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
GRADIENT_ACCUMULATION_STEPS = None
```

### Current Configuration (for 4GB GPU)
```python
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
GRADIENT_ACCUMULATION_STEPS = 4
```

**Effective batch size:** 4 × 4 = 16 (same as original)

---

## My Hardware Specifications

### Current Setup (Pop!_OS Laptop)
- **GPU:** 3.63 GB total memory
- **CUDA:** Available
- **System:** Pop!_OS (Linux)
- **Python:** 3.10

### Home Desktop (Windows)
- **GPU:** RTX 3090 (24GB)
- **System:** Windows
- **Python:** 3.10

---

## Performance Expectations

### With Current Config (3.6GB GPU):
- **Data Loading:** ~10 seconds
- **DistilBERT Fine-tuning:** ~45-90 minutes
- **Base Model Evaluation:** ~5-10 minutes each
- **Classical ML Training:** ~1-2 minutes
- **Total Runtime:** ~1-2 hours

### If Using CPU (fallback):
- **Total Runtime:** ~4-8 hours

---

## Recommended Hardware for Future Runs
- **Minimum:** 6GB GPU (GTX 1060, RTX 3050)
- **Recommended:** 8GB+ GPU (RTX 3060, RTX 4060)
- **Ideal:** 12GB+ GPU (RTX 3080, RTX 4070)

---

## Dependencies Issues (if any)
- `torch` - ✅ Working
- `transformers` - ✅ Working
- `sklearn` - ✅ Working (with warnings)
- `matplotlib` - ⚠️  Display issues
- `pandas` - ✅ Working
- `seaborn` - ⚠️  Related to matplotlib

---

## Recovery Strategy After Crash

**What Happened:** My initial run took 3 hours to train, then crashed during the plotting phase.

**Good News:** The fine-tuned DistilBERT model was successfully saved to `models/finetuned_distilbert/` before the crash!

**My Solution:** I created `complete_analysis.py` script that:
1. Loads my already-trained fine-tuned model
2. Only trains the fast models (base models load instantly, LR trains in ~2 min)
3. Performs all evaluations and generates visualizations
4. **Total time: 5-10 minutes instead of 3 hours**

**Files I Created:**
- `complete_analysis.py` - Recovery script to complete my analysis

**Status:** ✅ Implemented

---

## Jupyter Notebook Optimization for My RTX 3090

**Changes I Made:**
1. Added matplotlib backend fix (`matplotlib.use('Agg')`)
2. Optimized hyperparameters for my RTX 3090 at home:
   - `BATCH_SIZE = 32` (8x larger than what I had to use on my small GPU)
   - `MAX_LENGTH = 512` (full length)
   - Added configuration comments so I can easily switch between GPUs
3. Notebook now works on both my small GPU (4GB on Pop!_OS) and my gaming PC (24GB RTX 3090)

**Expected Performance on My RTX 3090:**
- Training time: ~20-30 minutes (vs 3 hours on my small GPU!)
- Better accuracy with full 512 token length
- 8x faster with batch size 32

**Status:** ✅ Ready to run on my Windows desktop

---

---

## RTX 3090 Results (Windows Desktop)
**Date:** 2025-11-01

### Configuration Used
```python
BATCH_SIZE = 32  # 8x larger than laptop
EPOCHS = 4  # Extra epoch for better accuracy
LEARNING_RATE = 2e-5
MAX_LENGTH = 512  # Full sequence length
GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation needed
```

### Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | Rank |
|-------|----------|-----------|--------|----------|------|
| **Fine-tuned DistilBERT** | **93.69%** | **93.18%** | **94.28%** | **93.73%** | 🥇 |
| Logistic Regression | 89.89% | 89.42% | 90.48% | 89.95% | 🥈 |
| Base GPT-2 | 50.02% | 50.01% | 99.50% | 66.56% | 🥉 |
| Base DistilBERT | 50.00% | 50.00% | 0.08% | 0.16% | 4 |

### Key Findings
- **Fine-tuned DistilBERT achieved 93.73% F1-Score** - excellent performance!
- Training time: ~25-35 minutes (vs 3+ hours on laptop)
- 8x faster training with batch size 32 vs 4
- Full 512 token sequences improved accuracy vs 256 tokens
- Base models (without fine-tuning) perform poorly, showing the importance of domain-specific training

### Hardware Comparison

| Metric | Laptop (3.6GB GPU) | Desktop (RTX 3090) | Improvement |
|--------|-------------------|-------------------|-------------|
| Batch Size | 4 | 32 | **8x larger** |
| Max Length | 256 | 512 | **2x longer** |
| Batches/Epoch | 9,000 | 1,125 | **8x fewer** |
| Training Time | 3+ hours | ~30 min | **6x faster** |
| Total VRAM | 3.6 GB | 24 GB | **6.7x more** |

**Status:** ✅ Complete with excellent results!

---

## Next Steps
1. ~~Fix matplotlib backend issue (use Agg)~~ ✅ Done
2. ~~Optionally fix joblib warnings~~ ✅ Done
3. ~~Test full run with fixed configuration~~ ✅ Done (but crashed)
4. ~~Create recovery script~~ ✅ Done
5. ~~Run recovery script and verify all outputs~~ ✅ Done
6. ~~Optimize Jupyter notebook for RTX 3090~~ ✅ Done
7. ~~Push to GitHub and test on Windows desktop~~ ✅ Done
8. ~~Run on RTX 3090 with optimized settings~~ ✅ Done - 93.73% F1!
9. ~~Create and run AI test cases~~ ✅ Done - 100% accuracy!
10. ~~Answer analysis questions~~ ✅ Done - analysis_questions.md
11. ~~Document final results in report~~ ✅ Done

---

## Final Project Status

**Date Completed:** November 1, 2025  
**Status:** ✅ COMPLETE - Ran everything 

### Deliverables Checklist (400/400 points) hopefully

- [x] Code: `imdb_analysis.py`, `run_testcases.py`, `retrain_lr.py`
- [x] Test Cases: `testcases.md` with GAICO format and actual results
- [x] Analysis Report: `analysis_questions.md` with all 5 questions answered
- [x] Results: All plots, confusion matrices, performance CSVs generated
- [x] Models: Fine-tuned DistilBERT and Logistic Regression saved
- [x] Documentation: README.md and this NOTES.md file

### Key Achievements

1. **93.73% F1-Score** with fine-tuned DistilBERT on RTX 3090
2. **100% test case accuracy** for both fine-tuned DistilBERT and Logistic Regression
3. **8x faster training** compared to laptop (30 min vs 3+ hours)
4. **Surprising finding**: Logistic Regression only 3.78% behind transformer
5. **Base models completely failed** (0.16% F1) - proves importance of fine-tuning

### Files Generated

**Code:**
- `imdb_analysis.py` - Main training and evaluation script
- `run_testcases.py` - AI test case evaluation
- `retrain_lr.py` - Quick LR retraining script

**Results:**
- `results/loss_accuracy_curves.png`
- `results/confusion_matrices.png`
- `results/metrics_comparison.png`
- `results/performance_comparison.csv`
- `results/testcase_results.csv`

**Models:**
- `models/finetuned_distilbert/` (fine-tuned transformer)
- `models/lr_model.pkl` (logistic regression)
- `models/tfidf_vectorizer.pkl` (feature extractor)

**Documentation:**
- `testcases.md` - 3 test cases in GAICO format with results
- `analysis_questions.md` - Comprehensive answers to 5 required questions
- `NOTES.md` - This development log
- `README.md` - Project overview and instructions

---

## Notes
- Training loss and validation metrics are being calculated correctly
- Models are being saved properly
- RTX 3090 configuration delivers excellent results with 93.73% F1-Score
- The difference between laptop and desktop is massive - 8x faster, better accuracy
- Test cases revealed surprising weakness in base GPT-2 (failed easiest test!)
- Logistic Regression's strong performance shows classical ML is still competitive
