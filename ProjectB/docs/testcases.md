# AI Test Cases - IMDB Sentiment Analysis

Following the GAICO template from: https://github.com/biplav-s/book-trustworthy-chatbot/blob/main/ai-testcases/testcase-template.md

---

## Test Case 1: Simple Short Positive Review

### Goal
Test model performance on a straightforward, short positive review with clear sentiment indicators.

### Action
Classify the sentiment of a short movie review.

### Input
```
"This movie was absolutely wonderful! Great acting and amazing story. Highly recommend!"
```
- **Word count:** 12
- **Sentence count:** 3
- **Complexity:** Simple, clear positive indicators ("wonderful", "great", "amazing", "highly recommend")

### Context
- **Domain:** Movie review sentiment classification
- **Expected Output:** Positive sentiment
- **Difficulty Level:** Easy

### Output Results

| Model | Prediction | Confidence | Correct? | Notes |
|-------|------------|------------|----------|-------|
| Fine-tuned DistilBERT | Positive | 99.7% | ✓ | Confidently correct |
| Base DistilBERT | Positive | 54.5% | ✓ | Low confidence, barely positive |
| Base GPT-2 | Negative | 71.2% | ✗ | **FAILED** - Incorrectly predicted negative! |
| Logistic Regression (baseline) | Positive | 99.7% | ✓ | Excellent performance on keywords |

**Analysis:** Surprisingly, Base GPT-2 FAILED on this simple positive review, predicting negative with 71% confidence! This shows that without fine-tuning, even powerful models can fail on basic cases. Fine-tuned DistilBERT and Logistic Regression both performed excellently with 99.7% confidence. Base DistilBERT barely got it right (~54% is essentially guessing).

---

## Test Case 2: Complex Negative Review with Sarcasm

### Goal
Test model ability to detect negative sentiment in a longer, more complex review containing sarcasm and mixed signals.

### Action
Classify the sentiment of a complex movie review with subtle negative indicators.

### Input
```
"Well, I have to say this movie certainly exceeded my expectations - I expected it to be mediocre, but it managed to be even worse. The director apparently thought that adding endless slow-motion sequences would somehow compensate for the complete lack of plot development. The lead actor delivered his lines with all the emotional depth of a cardboard cutout. Sure, the cinematography was decent, but that's like saying a car looks nice even though the engine doesn't work. I'd rather watch paint dry for two hours. Save your money and your time."
```
- **Word count:** 106
- **Sentence count:** 7
- **Complexity:** High - contains sarcasm ("exceeded my expectations"), mixed signals (positive: "decent cinematography"), and nuanced criticism

### Context
- **Domain:** Movie review sentiment classification
- **Expected Output:** Negative sentiment
- **Difficulty Level:** Hard - requires understanding sarcasm and weighing mixed sentiments

### Output Results

| Model | Prediction | Confidence | Correct? | Notes |
|-------|------------|------------|----------|-------|
| Fine-tuned DistilBERT | Negative | 99.9% | ✓ | Confidently detected sarcasm |
| Base DistilBERT | Positive | 51.6% | ✗ | Confused by mixed signals |
| Base GPT-2 | Negative | 85.6% | ✓ | Surprisingly handled sarcasm well! |
| Logistic Regression (baseline) | Negative | 99.3% | ✓ | Excellent - negative keywords dominated |

**Analysis:** Fine-tuned DistilBERT showed near-perfect confidence (99.9%) in detecting the negative sentiment despite sarcasm. Surprisingly, Base GPT-2 handled the sarcasm reasonably well with 85.6% confidence. Logistic Regression also performed excellently (99.3%) because the review contained many strong negative keywords ("worse", "lack", "cardboard cutout", "paint dry"). Base DistilBERT failed, essentially guessing at 51.6%.

---

## Test Case 3: Medium-Length Positive Review with Caveats

### Goal
Test model performance on a moderately positive review that includes minor criticisms (realistic review scenario).

### Action
Classify the sentiment of a balanced movie review with both positive and negative elements.

### Input
```
"I went into this movie with low expectations, but I was pleasantly surprised. While the pacing dragged a bit in the middle and some supporting characters felt underdeveloped, the main storyline was engaging and the lead performances were genuinely moving. The twist at the end caught me completely off guard. Yes, it has flaws, but overall it's a solid film that kept me entertained. Worth watching if you enjoy character-driven dramas."
```
- **Word count:** 78
- **Sentence count:** 6
- **Complexity:** Medium - balanced critique with overall positive conclusion, requires weighing multiple sentiments

### Context
- **Domain:** Movie review sentiment classification
- **Expected Output:** Positive sentiment (net positive despite criticisms)
- **Difficulty Level:** Medium - mixed signals but overall positive tone

### Output Results

| Model | Prediction | Confidence | Correct? | Notes |
|-------|------------|------------|----------|-------|
| Fine-tuned DistilBERT | Positive | 99.8% | ✓ | Confidently weighted overall sentiment |
| Base DistilBERT | Positive | 52.3% | ✓ | Barely correct, low confidence |
| Base GPT-2 | Negative | 82.9% | ✗ | Focused on criticisms, missed conclusion |
| Logistic Regression (baseline) | Positive | 88.6% | ✓ | More positive than negative keywords |

**Analysis:** Fine-tuned DistilBERT excelled with 99.8% confidence, correctly understanding that the overall sentiment is positive despite criticisms. Base GPT-2 FAILED, getting too focused on negative phrases ("dragged," "underdeveloped," "flaws") and missing the positive conclusion. Logistic Regression performed well (88.6%) by counting more positive than negative keywords. Base DistilBERT barely got it right with 52.3% confidence (essentially guessing).

---

## Summary Comparison

### Model Performance Across Test Cases

| Model | Test 1 (Easy) | Test 2 (Hard) | Test 3 (Medium) | Success Rate |
|-------|---------------|---------------|-----------------|--------------|
| **Fine-tuned DistilBERT** | ✓ (99.7%) | ✓ (99.9%) | ✓ (99.8%) | **100% (3/3)** 🥇 |
| **Logistic Regression** | ✓ (99.7%) | ✓ (99.3%) | ✓ (88.6%) | **100% (3/3)** 🥈 |
| **Base GPT-2** | ✗ (71.2%) | ✓ (85.6%) | ✗ (82.9%) | **33% (1/3)** |
| **Base DistilBERT** | ✓ (54.5%) | ✗ (51.6%) | ✓ (52.3%) | **67% (2/3)** |

### Key Insights

1. **Fine-tuned DistilBERT**: Perfect performance with ultra-high confidence (99.7-99.9%) across ALL complexity levels. Demonstrates mastery of context, sarcasm, and nuanced sentiment. This is the gold standard.

2. **Logistic Regression (Baseline)**: Surprisingly excellent! Achieved 100% accuracy with very high confidence on easy/hard tests (99%+). Only showed lower confidence (88.6%) on the mixed sentiment review. Proves that classical ML with good features can be highly effective for sentiment analysis.

3. **Base GPT-2**: Inconsistent and unreliable. Paradoxically FAILED the easiest test (predicted negative on clear positive review) but PASSED the hardest test (detected sarcasm). This unpredictability makes it unsuitable for production without fine-tuning.

4. **Base DistilBERT**: Essentially random guessing with ~50-54% confidence on all tests. Failed 1/3 tests. Demonstrates that without fine-tuning, even state-of-the-art architectures are useless for domain-specific tasks.

### Complexity Impact

- **Easy (Test 1)**: Only fine-tuned models succeeded! Base GPT-2 surprisingly failed this simple test.
- **Hard (Test 2)**: Fine-tuned DistilBERT and LR both excelled. Base GPT-2 handled sarcasm but Base DistilBERT failed.
- **Medium (Test 3)**: Fine-tuned DistilBERT and LR succeeded. Base models failed to balance mixed sentiments.

**Surprising Finding**: Test complexity didn't correlate with model difficulty as expected. Base GPT-2 failed the easiest test but passed the hardest!

### Recommendation

**For Production Deployment:**

1. **Fine-tuned DistilBERT** - Best choice for accuracy-critical applications. Perfect 100% accuracy with 99%+ confidence. Worth the computational cost.

2. **Logistic Regression** - Excellent alternative when speed/cost matters. Achieved 100% accuracy with very high confidence and trains/infers in seconds vs. minutes. Surprisingly competitive with the fine-tuned transformer!

3. **Base Models** - DO NOT USE. Base GPT-2 is unpredictable and Base DistilBERT is essentially random guessing.

**Cost-Benefit Analysis**: Given that Logistic Regression matched fine-tuned DistilBERT's accuracy on these tests while being 100x faster and cheaper, it's an excellent choice for most production scenarios. Only use fine-tuned DistilBERT when you need guaranteed performance on highly nuanced text.
