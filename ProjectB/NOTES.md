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

## Next Steps
1. ~~Fix matplotlib backend issue (use Agg)~~ ✅ Done
2. ~~Optionally fix joblib warnings~~ ✅ Done
3. ~~Test full run with fixed configuration~~ ✅ Done (but crashed)
4. ~~Create recovery script~~ ✅ Done
5. ~~Run recovery script and verify all outputs~~ ✅ Done
6. ~~Optimize Jupyter notebook for RTX 3090~~ ✅ Done
7. Push to GitHub and test on Windows desktop
8. Document final results in report

---

## Notes
- Training loss and validation metrics are being calculated correctly
- Models are being saved properly
- The crashes happen during visualization phase, not training
- All computation (training/evaluation) completed successfully before crash
