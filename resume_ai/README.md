# Resume AI Analysis Project

A comprehensive Python-based system for analyzing and comparing resumes using natural language processing, keyword extraction, and similarity analysis.

## 📋 Project Overview

This project processes PDF resumes to extract meaningful insights through:
- **Keyword Analysis**: Identifies relevant technical and professional keywords
- **Resume Classification**: Categorizes resumes based on content and skills
- **Similarity Analysis**: Compares resumes using cosine similarity metrics
- **Text Processing**: Extracts and cleans text from PDF files
- **Statistical Reports**: Generates detailed CSV outputs for analysis

## 🏗️ Project Structure

```
resume_ai/
├── README.md                          # This file
├── keywords_enhanced.py               # Enhanced analysis pipeline (authoritative)
├── resume_classifier_enhanced.py      # Enhanced classification pipeline
├── resume_classifier.py               # Legacy wrapper → forwards to enhanced
├── resume_keywords.csv                # Curated CS/CE keyword taxonomy
├── StudentResumes/                    # Collection of student resumes
│   ├── Aashish_Jayapuram_Resume_Revised.pdf
│   ├── AidanVanVoorhis_2025_Resume.pdf
│   ├── Ardoine_Docteur_Resume.pdf
│   ├── [... additional student resumes]
│   └── TylerKorth_Resume_AI_Class.pdf
└── out/                               # Generated analysis outputs
    ├── resume_texts_enhanced.csv      # Extracted text + metadata + degree fields
    ├── resume_word_counts_enhanced.csv# Curated BOW features
    ├── resume_keyword_scores_enhanced.csv # Keyword scores (with degree and taxonomy)
    └── resume_similarity_enhanced.csv # Cosine similarity matrix
```

## 🚀 Features

### 1. **Keyword Analysis (Enhanced)** (`keywords_enhanced.py`)
- Extracts text from PDF/DOCX/TXT/ODS
- Adds structured metadata (emails, phones, LinkedIn, GitHub, file stats)
- Detects degree level (bachelor’s/master’s) and field (CS/CE)
- Scores resumes with curated CS/CE keyword taxonomy (good/bad)
- Generates comprehensive enhanced reports


### 2. **Resume Classification (Enhanced)** (`resume_classifier_enhanced.py`)
- Multiple models, cross‑validation, feature engineering
- Real data loading and robust evaluation

Legacy entry point `resume_classifier.py` now forwards to `resume_classifier_enhanced.py`.

### 3. **Similarity Analysis**
- Computes cosine similarity between resumes
- Identifies similar candidates based on skills and experience
- Creates similarity matrices for comparative analysis

### 4. **Text Processing & Analysis**
- PDF text extraction using PyPDF2/pdfplumber
- Text cleaning and preprocessing
- Word frequency analysis and statistics

## 🔧 Technical Requirements

### Dependencies
The project utilizes the **pyexcel-ods3** library for reading ODS files (as specified in user rules).

```python
# Key libraries used:
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from docx import Document  # python-docx
import pyexcel_ods3  # ODS support (per user rule)
import nltk  # stopwords, tokenization
# Optional: spaCy (if installed) for advanced NLP
```

### Installation
```bash
pip install pandas numpy scikit-learn pdfplumber python-docx pyexcel-ods3 nltk
# Optional for advanced NLP:
# pip install spacy && python -m spacy download en_core_web_sm
```

## 📊 Output Files

### CSV Reports Generated (Enhanced):

1. **`resume_texts_enhanced.csv`**
   - filename, label, text
   - Metadata: file_size, file_type, extraction_method
   - Contact/profile: emails, phones, urls, linkedin, github
   - Education: degree_level (bachelors/masters/phd/unknown), degree_field (computer_science/computer_engineering/empty)
   - Quality: char_count, word_count, line_count

2. **`resume_word_counts_enhanced.csv`**
   - Bag-of-words features with curated, de-noised tokens suitable for modeling

3. **`resume_keyword_scores_enhanced.csv`**
   - filename, label, degree_level, degree_field
   - good_hits, bad_hits, quality_score
   - good_terms_found, bad_terms_found
   - word_count, char_count
   - Per-category counts for: Programming, Data Science, Cloud/DevOps, Tools, Soft Skills,
     Certifications, Experience Level, Impact Words, Computer Science Core, Computer Engineering Core

4. **`resume_similarity_enhanced.csv`**
   - Cosine similarity matrix across resumes (0 to 1)

## 🎯 Use Cases

### Academic Research
- Analyze trends in student resumes and skill development
- Compare technical skills across different academic programs
- Study the evolution of resume content over time

### Recruitment & HR
- Screen resumes based on keyword matching
- Identify similar candidates for position clustering
- Analyze skill gaps in candidate pools

### Career Development
- Benchmark individual resumes against peer groups
- Identify missing keywords for specific industries
- Optimize resume content based on similarity analysis

## 📈 Analysis Insights

The system provides insights into:
- **Technical Skills Distribution**: Programming languages, tools, frameworks
- **Experience Levels**: Junior, senior, leadership roles
- **Industry Focus**: Software development, AI/ML, web development
- **Educational Background**: Degree types, institutions, coursework
- **Professional Keywords**: Project management, collaboration, problem-solving

## 🔍 Sample Analysis Results

Based on the current dataset:
- **Most Common Skills**: Java, Python, JavaScript, Linux, Git
- **Emerging Technologies**: AI/ML, Docker, cloud platforms
- **Leadership Indicators**: Manager, lead, senior positions
- **Educational Trends**: Computer Science, Engineering majors

## 🚀 Running the Analysis

### Basic Usage:
```bash
# Run enhanced keyword analysis (writes resume_keywords.csv if missing)
python keywords_enhanced.py --root StudentResumes --outdir out --keywords resume_keywords.csv

# Run enhanced classification on the enhanced texts
python resume_classifier_enhanced.py --resume-data out/resume_texts_enhanced.csv --output-dir classification_results

# (Optional) If the legacy wrapper exists, it forwards to the enhanced script
# python resume_classifier.py --resume-data out/resume_texts_enhanced.csv --output-dir classification_results
```

### Customization:
- Update `resume_keywords.csv` to modify keyword matching criteria
- Adjust similarity thresholds in the analysis scripts
- Add new resume categories for classification

### Keyword taxonomy CSV (resume_keywords.csv)
A simple CSV with three columns: Category, Good_Keywords, Bad_Keywords. Terms are comma-separated within each cell. Example:

```csv path=null start=null
Category,Good_Keywords,Bad_Keywords
Programming,"python, java, c, c++, c#, javascript, typescript, go, rust, kotlin, swift, matlab, r","basic computer skills, microsoft office only, beginner"
Data Science,"machine learning, deep learning, nlp, ai, pandas, numpy, scikit-learn, tensorflow, pytorch, statistics, data mining, computer vision","some experience with data, basic statistics"
Cloud/DevOps,"aws, azure, gcp, docker, kubernetes, ci/cd, terraform, ansible, jenkins, linux, bash","familiar with cloud, tinkered with aws"
Computer Science Core,"data structures, algorithms, operating systems, computer architecture, discrete mathematics, databases, networking, software engineering, object-oriented programming, compilers","basic computer skills"
Computer Engineering Core,"embedded systems, digital logic, verilog, vhdl, fpga, microcontroller, microcontrollers, microprocessor, systemverilog, pcb design, circuits, hardware design, signal processing, control systems","breadboard only, simple circuits only"
```

## 📝 Notes

- **Degree detection**: Uses strict patterns to avoid false positives (e.g., ignoring "MS Word").
  - Detects PhD (PhD/Ph.D./Doctor of Philosophy), Masters (M.S./MSc/MEng/M.E./Master of …),
    and Bachelors (B.S./BSc/BEng/B.E./Bachelor of …), including common OCR artifacts
    like smashed strings (e.g., "BachelorofScience").
  - Some resumes may remain `unknown` if PDFs are image-heavy or degree lines don’t extract cleanly.
- **PDF Processing**: Handles various PDF formats and layouts; text quality depends on extraction.
- **Text Cleaning**: Removes formatting artifacts and normalizes text.
- **NLTK**: Required data is auto-downloaded on first run (punkt, stopwords, wordnet).
- **spaCy (optional)**: If installed, will be utilized for advanced NLP.
- **Scalability**: Designed to handle large resume datasets.
- **Flexibility**: Easy to extend with new analysis features.

## 🔮 Future Enhancements

- Integration with job posting analysis
- Real-time resume scoring API
- Machine learning model for automatic keyword extraction
- Advanced NLP techniques for semantic similarity
- Dashboard visualization for analysis results

## 👥 Contributors

- Pedro Fischetti - Project Developer and Maintainer

## 📄 License

This project is developed for academic and research purposes as part of coursework at the University of South Carolina.

---

*Last Updated: September 2025*
