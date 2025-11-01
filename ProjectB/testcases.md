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
| Fine-tuned DistilBERT | Positive | 99.8% | ✓ | Confidently correct |
| Base DistilBERT | Positive | 52.1% | ✓ | Low confidence, barely positive |
| Base GPT-2 | Positive | 98.9% | ✓ | High confidence |
| Logistic Regression (baseline) | Positive | 95.3% | ✓ | Strong performance on keywords |

**Analysis:** All models correctly identified this simple positive review. Fine-tuned DistilBERT showed the highest confidence. Even base models performed well due to clear sentiment words.

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
| Fine-tuned DistilBERT | Negative | 99.1% | ✓ | Correctly detected despite sarcasm |
| Base DistilBERT | Positive | 51.2% | ✗ | Confused by mixed signals |
| Base GPT-2 | Positive | 87.3% | ✗ | Fooled by "decent", "nice" keywords |
| Logistic Regression (baseline) | Negative | 78.4% | ✓ | TF-IDF captured negative keywords |

**Analysis:** Fine-tuned DistilBERT excelled at understanding context and sarcasm. Base models struggled with nuance, being misled by positive words like "decent" and "nice". This demonstrates the value of fine-tuning for complex sentiment analysis.

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
| Fine-tuned DistilBERT | Positive | 92.7% | ✓ | Correctly weighted overall sentiment |
| Base DistilBERT | Negative | 53.8% | ✗ | Focused too much on negative phrases |
| Base GPT-2 | Positive | 68.5% | ✓ | Detected positive conclusion |
| Logistic Regression (baseline) | Positive | 71.2% | ✓ | More positive than negative keywords |

**Analysis:** Fine-tuned DistilBERT best understood that despite minor criticisms, the review is ultimately positive. Base DistilBERT was misled by negative phrases like "dragged," "underdeveloped," and "flaws." This shows fine-tuning improves contextual understanding.

---

## Summary Comparison

### Model Performance Across Test Cases

| Model | Test 1 (Easy) | Test 2 (Hard) | Test 3 (Medium) | Success Rate |
|-------|---------------|---------------|-----------------|--------------|
| **Fine-tuned DistilBERT** | ✓ (99.8%) | ✓ (99.1%) | ✓ (92.7%) | **100% (3/3)** |
| **Base DistilBERT** | ✓ (52.1%) | ✗ (51.2%) | ✗ (53.8%) | **33% (1/3)** |
| **Base GPT-2** | ✓ (98.9%) | ✗ (87.3%) | ✓ (68.5%) | **67% (2/3)** |
| **Logistic Regression** | ✓ (95.3%) | ✓ (78.4%) | ✓ (71.2%) | **100% (3/3)** |

### Key Insights

1. **Fine-tuned DistilBERT**: Best overall performance with high confidence across all complexity levels. Excels at understanding context, sarcasm, and nuanced sentiment.

2. **Base DistilBERT**: Performs poorly without fine-tuning. Low confidence scores (~50-53%) indicate it's essentially guessing. Demonstrates the critical importance of domain-specific fine-tuning.

3. **Base GPT-2**: Decent performance on simple cases but struggles with sarcasm and mixed sentiments. Better than base DistilBERT but not reliable for complex reviews.

4. **Logistic Regression (Baseline)**: Surprisingly robust performance! While confidence is lower than fine-tuned DistilBERT, it correctly classified all test cases by counting sentiment keywords. However, this approach would fail on more subtle sentiment expressions.

### Complexity Impact

- **Easy (Test 1)**: All models succeeded - clear sentiment words are sufficient
- **Medium (Test 3)**: Fine-tuning advantage emerges for balanced reviews
- **Hard (Test 2)**: Only fine-tuned model consistently handles sarcasm and nuance

### Recommendation

For production deployment, **Fine-tuned DistilBERT** is the clear winner for complex sentiment analysis, while **Logistic Regression** could be acceptable for simple reviews where speed matters more than subtle accuracy.
