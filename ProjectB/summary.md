# ProjectB Summary – IMDB Sentiment Analysis (CSCE 580 Project B)

## 1. Goal of the Project

This project builds and evaluates an end‑to‑end **sentiment analysis** system on the **IMDB movie review dataset**.  

The main objective is to **compare modern large language models (LLMs)** with a **classical machine learning baseline** for binary sentiment classification (positive vs. negative):

- **Fine‑tuned DistilBERT** (transformer, trained on IMDB)
- **Base DistilBERT** (pre‑trained only, no fine‑tuning)
- **Base GPT‑2** (pre‑trained only, no fine‑tuning)
- **Logistic Regression + TF‑IDF** (classical ML baseline)

The project does not just report accuracy; it also analyzes **training curves, confusion matrices, test cases, and deployment trade‑offs** (accuracy vs. speed vs. cost).

---

## 2. Data and Task

- Dataset file: `data/IMDB Dataset.csv`
- Each row:  
  - `review` – full free‑text review  
  - `sentiment` – `"positive"` or `"negative"`
- Labels are converted to:
  - `1` = positive  
  - `0` = negative
- Splits:
  - Train / validation / test using `train_test_split` with stratification:
    - ~80% train (then split again into train/val)
    - ~20% test
  - Stratification keeps class balance consistent across splits.

Task: **Binary sentiment classification** – predict positive or negative sentiment given a review.

---

## 3. Project Structure

Key components:

- **Code**
  - `notebooks/imdb_sentiment_analysis.ipynb` – main exploratory notebook.
  - `src/imdb_analysis.py` – main script: full pipeline (training + evaluation + plots).
  - `src/complete_analysis.py` – runs all evaluations and plots assuming the fine‑tuned model already exists (no retraining).
  - `src/run_testcases.py` – runs three carefully designed test cases across all four models and saves results.

- **Models**
  - `models/finetuned_distilbert/` – saved fine‑tuned DistilBERT weights.
  - `models/lr_model.pkl` – trained Logistic Regression model.
  - `models/tfidf_vectorizer.pkl` – fitted TF‑IDF vectorizer.

- **Results**
  - `results/loss_accuracy_curves.png` – training vs validation loss/accuracy over epochs.
  - `results/confusion_matrices.png` – confusion matrices for all four models.
  - `results/metrics_comparison.png` – bar chart of metrics (Accuracy/Precision/Recall/F1).
  - `results/performance_comparison.csv` – tabular metrics comparison.
  - `results/testcase_results.csv` – detailed outputs for the three GAICO test cases.

- **Documentation**
  - `README.md` – high‑level overview, how to run, and summarized results.
  - `docs/analysis_questions.md` – detailed answers to the five analysis questions.
  - `docs/testcases.md` – GAICO‑formatted AI test cases and discussion.
  - `docs/NOTES.md` – development log and troubleshooting notes.

---

## 4. Implementation Overview

### 4.1 Common Data Pipeline

Implemented mainly in `src/imdb_analysis.py`:

1. **Load and preprocess data**
   - Read `IMDB Dataset.csv` with pandas.
   - Map `"positive"`/`"negative"` to `1`/`0`.
2. **Train/validation/test split**
   - Use stratified `train_test_split` to maintain label balance.
3. **Return text and label arrays** for all three splits.

### 4.2 Transformer Models (DistilBERT & GPT‑2)

**Dataset & Tokenization**

- Custom PyTorch `Dataset` (`IMDBDataset`) that:
  - Takes raw text and labels.
  - Uses a tokenizer (DistilBERT or GPT‑2) to produce `input_ids` and `attention_mask`.
  - Pads/truncates to a fixed `MAX_LENGTH` (512 in the main script).
- Data is loaded via `DataLoader`s for train/val/test with `batch_size` tuned for GPU.

**Fine‑tuned DistilBERT**

- Uses `DistilBertForSequenceClassification` with `num_labels=2`.
- Training:
  - Optimizer: `AdamW`.
  - Scheduler: `get_linear_schedule_with_warmup`.
  - `EPOCHS = 4` with gradient clipping.
  - For each epoch: run `train_epoch` then `eval_model` to track:
    - Training/validation loss.
    - Training/validation accuracy.
- After training:
  - Model is saved into `models/finetuned_distilbert/`.
  - Training stats are later used to plot loss/accuracy curves.

**Base DistilBERT and Base GPT‑2**

- Loaded with sequence classification heads, but **not fine‑tuned on IMDB**:
  - Base DistilBERT: `distilbert-base-uncased`.
  - Base GPT‑2: `gpt2` with classification head and EOS as pad token.
- These models are evaluated to show how poorly un‑fine‑tuned models perform on a domain‑specific task.

### 4.3 Classical Model (Logistic Regression + TF‑IDF)

**Feature Extraction**

- Use `TfidfVectorizer(max_features=10000, ngram_range=(1, 2))`:
  - Fit on training text.
  - Transform validation and test text into sparse TF‑IDF matrices.

**Model**

- `LogisticRegression(max_iter=1000, n_jobs=1)`:
  - Trained on TF‑IDF features and labels.
  - Quick validation accuracy check on the validation set.

**Saving**

- Saves:
  - `models/lr_model.pkl` – trained classifier.
  - `models/tfidf_vectorizer.pkl` – vectorizer to reproduce features later.

### 4.4 Evaluation & Visualizations

**Predictions**

- Transformers:
  - `get_predictions()` loops over a `DataLoader` and collects predicted labels vs. true labels.
- Logistic Regression:
  - Directly uses `.predict()` on TF‑IDF test features.

**Metrics**

- For each model:
  - Accuracy, precision, recall, F1 (binary).
- Aggregated into a DataFrame and saved as:
  - `results/performance_comparison.csv`.
- A printed summary shows rankings by F1‑Score.

**Confusion Matrices**

- One confusion matrix per model.
- Visualized as heatmaps with seaborn:
  - Saved as `results/confusion_matrices.png`.
- Used to analyze error types (false positives vs false negatives, class bias).

**Training Curves**

- Uses recorded `train_loss`, `val_loss`, `train_acc`, `val_acc` from fine‑tuning:
  - Plots loss/accuracy vs epoch.
  - Saved as `results/loss_accuracy_curves.png`.
- These curves are analyzed in `docs/analysis_questions.md` to reason about overfitting, learning rate, and generalization.

**Metric Comparison Chart** (in `complete_analysis.py`)

- Bar chart comparing Accuracy/Precision/Recall/F1 across all four models:
  - Saved as `results/metrics_comparison.png`.

---

## 5. GAICO Test Cases and Behavioral Analysis

Beyond aggregate metrics, the project includes **three targeted AI test cases** described in `docs/testcases.md` and executed by `src/run_testcases.py`:

1. **Test Case 1 – Simple short positive review**
   - Very clear, strongly positive phrases.
2. **Test Case 2 – Complex negative review with sarcasm**
   - Long, sarcastic, mixed signals but overall negative.
3. **Test Case 3 – Medium-length positive review with caveats**
   - Contains both positive and negative aspects, but overall positive.

`run_testcases.py`:

- Loads:
  - Fine‑tuned DistilBERT.
  - Base DistilBERT.
  - Base GPT‑2.
  - Logistic Regression + TF‑IDF (from saved `.pkl` files).
- For each test case and each model:
  - Produces a **prediction** and a **confidence score**.
- Results:
  - Printed to the console.
  - Saved as `results/testcase_results.csv`.
  - Summarized by success rate and average confidence per model.

These test cases highlight **behavioral differences**:
- Fine‑tuned DistilBERT and Logistic Regression both achieve **100% accuracy** on the three test cases, with very high confidence.
- Base GPT‑2 and base DistilBERT are inconsistent/unreliable:
  - GPT‑2 sometimes fails easy cases and passes harder, sarcastic ones.
  - Base DistilBERT is close to random guessing with ~50–54% confidence.

---

## 6. Key Results

From `README.md` and analysis:

Approximate performance on the test set:

| Model                 | Accuracy | F1‑Score | Training Time       |
|-----------------------|----------|---------:|---------------------|
| Fine‑tuned DistilBERT | 93.69%   | 93.73%   | ~30 minutes (GPU)   |
| Logistic Regression   | 89.89%   | 89.95%   | ~2 minutes          |
| Base GPT‑2            | 50.02%   | 66.56%   | 0 (pre‑trained)     |
| Base DistilBERT       | 50.00%   | 0.16%    | 0 (pre‑trained)     |

Key observations:

- **Fine‑tuned DistilBERT**:
  - Best overall performance.
  - Training & validation curves show no overfitting and smooth convergence.
  - Requires **significant compute** (especially without a large GPU).

- **Logistic Regression (TF‑IDF)**:
  - Surprisingly competitive: only ~3.8 percentage points lower F1 than DistilBERT.
  - Training and inference are **orders of magnitude faster**.
  - Runs easily on CPU, small memory footprint, and high interpretability.

- **Base GPT‑2 and Base DistilBERT**:
  - Without task‑specific fine‑tuning, both are essentially unusable for IMDB sentiment:
    - Confusion matrices show extreme class biases.
    - Accuracies around 50% (like random guessing).
  - Demonstrates that **pre‑training alone is not enough** for this domain.

---

## 7. Conclusions and Deployment Recommendations

From `docs/analysis_questions.md`, the main conclusions are:

1. **Fine‑tuning is critical**  
   - Base DistilBERT goes from ~0.16% F1 to ~93.73% F1 after fine‑tuning.  
   - This ~93‑point jump shows that domain adaptation is essential.

2. **Classical ML is still very strong**  
   - Logistic Regression with TF‑IDF reaches ~90% F1, only ~3.8% behind the transformer.  
   - It is much faster and cheaper to train and deploy.

3. **Model choice depends on constraints**  
   - If **accuracy is absolutely critical** and resources are available → choose **fine‑tuned DistilBERT**.  
   - If **speed, cost, and simplicity** matter more → choose **Logistic Regression**.  
   - For many production scenarios, a **hybrid approach** is ideal:
     - First pass with Logistic Regression.
     - Only send low‑confidence or ambiguous cases to DistilBERT.

4. **Behavioral analysis via test cases is important**  
   - Aggregate metrics can hide weird behaviors (e.g., GPT‑2 failing the easiest test).
   - The GAICO test cases show how models behave on:
     - Very simple examples.
     - Sarcasm and mixed sentiment.
     - Realistic medium‑complexity reviews.

---

## 8. What This Project Demonstrates

Overall, ProjectB demonstrates:

- A complete **NLP workflow**: data preprocessing, model training, evaluation, visualization, and targeted test cases.
- A **quantitative and qualitative comparison** between LLM‑based models and a classical baseline.
- Practical insight into **when sophisticated transformers are worth the cost**, and when a simpler classical model is sufficient or even preferable in real‑world deployments.
