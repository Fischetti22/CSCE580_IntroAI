# GAICo Quick Reference Guide

## What is GAICo?
GAICo (Generative AI Comparison) is a Python library for comparing and evaluating outputs from different Large Language Models (LLMs). It's particularly useful for AI research, model evaluation, and ensuring AI alignment.

## Setup Instructions

### 1. Create Virtual Environment
```bash
python3 -m venv gaico-venv
source gaico-venv/bin/activate  # On Linux/Mac
# gaico-venv\Scripts\activate   # On Windows
```

### 2. Install GAICo
```bash
# Basic installation
pip install gaico

# Full installation with all features
pip install 'gaico[bertscore,cosine,jsd]'
```

### 3. Install Jupyter (optional, for running the notebook)
```bash
pip install jupyter
```

## Basic Usage

### Simple Example
```python
from gaico import Experiment

# Your model responses
llm_responses = {
    "Model A": "Response from model A",
    "Model B": "Response from model B", 
    "Model C": "Response from model C"
}

# Expected/ideal response
reference_answer = "The ideal response you want"

# Create experiment
exp = Experiment(
    llm_responses=llm_responses,
    reference_answer=reference_answer
)

# Get comparison results
results = exp.compare()
df = exp.to_dataframe()
print(df)
```

## Available Metrics

The results include various similarity metrics:

- **Jaccard**: Set-based similarity (0-1, higher = more similar)
- **Cosine**: Vector similarity using word embeddings
- **Levenshtein**: Edit distance between strings
- **BLEU**: Machine translation quality metric
- **ROUGE**: Text summarization evaluation metrics
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap  
  - ROUGE-L: Longest common subsequence
- **BERTScore**: BERT-based semantic similarity
  - Precision, Recall, F1
- **JSD**: Jensen-Shannon Divergence

## Key Methods

- `exp.compare()`: Run the comparison analysis
- `exp.to_dataframe()`: Convert results to pandas DataFrame
- `exp.summarize()`: Get summary statistics
- `exp.models`: List of model names being compared
- `exp.llm_responses`: Dictionary of model responses
- `exp.reference_answer`: The reference/ideal answer

## Understanding Results

### High Similarity Scores (> 0.5)
- Model response is very similar to reference
- Good alignment with expected behavior

### Medium Similarity Scores (0.1 - 0.5)
- Moderate similarity
- May need improvement

### Low Similarity Scores (< 0.1)
- Poor alignment with reference
- Significant deviation from expected response

## Running the Examples

### 1. Run the Demo Script
```bash
cd /path/to/gaico-usage
source gaico-venv/bin/activate
python complete_demo.py
```

### 2. Run the Jupyter Notebook
```bash
source gaico-venv/bin/activate
jupyter notebook LLMsOutputAnalysisWithGaico.ipynb
```

## Common Use Cases

1. **Model Comparison**: Compare how different AI models respond to the same prompt
2. **Safety Evaluation**: Check if models refuse inappropriate requests
3. **Quality Assessment**: Evaluate response quality against gold standards
4. **Research**: Analyze model behavior patterns
5. **Fine-tuning Evaluation**: Assess improvements after model training

## Tips for Success

- Use clear, specific reference answers
- Test with various types of prompts
- Pay attention to different metrics - they measure different aspects
- BERTScore is often most reliable for semantic similarity
- Jaccard is good for exact word overlap
- Use multiple metrics together for comprehensive evaluation

## Next Steps

1. Experiment with your own model responses
2. Try different types of prompts (factual, creative, safety-related)
3. Explore visualization features in the notebook
4. Read the full documentation: https://pypi.org/project/GAICo/
5. Consider using GAICo in your AI research projects

## Need Help?

- Check the existing `experiment_report.csv` to see expected output format
- Run the demo scripts to understand the workflow
- The library automatically handles tokenization and metric calculation
- Most errors are due to missing dependencies - ensure full installation
