#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path('/home/droski/Desktop/School/Fall25/AI/resume_ai/out')
DATA_CSV = BASE / 'resume_texts.csv'
REPORT_PATH = BASE / 'attention_report.txt'
RANKED_PATH = BASE / 'attention_scores.csv'

# Curated signals
CORE_LANGS = {
    'python','java','c','c++','c#','javascript','typescript','go','rust','ruby','kotlin','swift','matlab','r','sql'
}
FRAMEWORKS_LIBS = {
    'react','angular','vue','node','express','django','flask','spring','springboot','spring-boot','.net','dotnet',
    'pytorch','tensorflow','keras','scikit','scikit-learn','sklearn','numpy','pandas','opencv','matplotlib','seaborn',
    'xgboost','lightgbm','huggingface','transformers','fastapi','nextjs','next.js','django-rest','rest','grpc'
}
CLOUD_DEVOPS = {
    'aws','gcp','google cloud','azure','docker','kubernetes','k8s','terraform','linux','bash','shell',
    'jenkins','github actions','gitlab ci','cicd','ci/cd','prometheus','grafana','nginx','apache','airflow',
    'spark','hadoop','kafka','redis','rabbitmq','s3','ec2','lambda','cloudwatch','cloudformation'
}
DATABASES = {
    'postgres','postgresql','mysql','sqlite','mssql','sql server','mongodb','dynamodb','bigquery','redshift'
}
# Strong action verbs that signal impact
ACTION_VERBS = {
    'built','developed','designed','implemented','improved','optimized','automated','led','managed','delivered',
    'deployed','integrated','refactored','reduced','increased','achieved','collaborated','spearheaded','created',
    'analyzed','engineered','launched','maintained','migrated','scaled','debugged','tested','architected'
}
# Leadership/ownership cues
LEADERSHIP = {'led','managed','mentored','owned','directed','coordinated','supervised','organized'}

# Generic stop terms to downplay
GENERIC_STOP = {
    'university','college','education','coursework','gpa','semester','present','phone','email','linkedin',
    'address','city','state','zip','objective','references','responsible','assisted','helped'
}

# Regex for quantified claims (e.g., 20%, 2x, $50k, reduced 30%, increased by 3x)
QUANT_PATTERN = re.compile(r"(\b\d+(?:\.\d+)?\s*(?:%|percent|x|k|m|b)\b|\$\s*\d+[\d,]*|\b\d+\s*(?:users|customers|robots|merchants|datasets|projects|repos)\b)", re.I)

# Simple cleaner
def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Build vocabulary for TF-IDF restricted to skills/tools keywords
def build_vocab():
    vocab = set()
    for s in [CORE_LANGS, FRAMEWORKS_LIBS, CLOUD_DEVOPS, DATABASES]:
        vocab.update(s)
    # normalize dots and spaces
    normed = set()
    for term in vocab:
        normed.add(term.lower())
        normed.add(term.lower().replace('.', ''))
    return {v: i for i, v in enumerate(sorted(normed))}

VOCAB = build_vocab()

# Extract matches from text for sets
def find_terms(text: str, terms: set) -> set:
    found = set()
    for term in terms:
        pat = re.escape(term)
        if re.search(rf"\b{pat}\b", text):
            found.add(term)
    return found

# Tokenize simple words for action verbs
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z+#.\-/]*")

def tokenize_words(text: str):
    return WORD_RE.findall(text.lower())

# Score function
def attention_score(row):
    score = 0.0
    score += 2.0 * len(row['core_langs'])
    score += 1.6 * len(row['frameworks_libs'])
    score += 1.4 * len(row['cloud_devops'])
    score += 1.2 * len(row['databases'])
    score += 1.0 * len(row['action_verbs'])
    score += 0.8 * len(row['leadership'])
    score += 1.2 * row['quantified_claims']
    # Boost for TF-IDF weighted sum of tech terms
    score += 2.0 * row.get('tfidf_sum', 0.0)
    return score


def main():
    if not DATA_CSV.exists():
        raise SystemExit(f'Missing {DATA_CSV}')
    df = pd.read_csv(DATA_CSV)
    if 'text' not in df.columns:
        raise SystemExit('CSV must include a text column')
    if 'filename' not in df.columns:
        df['filename'] = [f'doc_{i}' for i in range(len(df))]

    texts = df['text'].astype(str).map(clean_text).tolist()

    # Restricted TF-IDF on skills/tools vocabulary
    vect = TfidfVectorizer(vocabulary=VOCAB, ngram_range=(1,1), token_pattern=r"(?u)\b\w[\w+.#/-]*\b")
    X = vect.fit_transform(texts)
    feature_names = np.array(vect.get_feature_names_out())

    rows = []
    for i, t in enumerate(texts):
        tech_vec = X[i]
        tfidf_sum = float(tech_vec.sum())
        # top tech terms
        if tech_vec.nnz:
            idx = np.argsort(tech_vec.toarray()[0])[::-1]
            top_terms = [feature_names[j] for j in idx[:12] if tech_vec.toarray()[0][j] > 0]
        else:
            top_terms = []

        # Extract sets
        core = find_terms(t, CORE_LANGS)
        fw = find_terms(t, FRAMEWORKS_LIBS)
        cloud = find_terms(t, CLOUD_DEVOPS)
        dbs = find_terms(t, DATABASES)
        tokens = tokenize_words(t)
        verbs = sorted(set([w for w in tokens if w in ACTION_VERBS]))
        leaders = sorted(set([w for w in tokens if w in LEADERSHIP]))
        quantified = len(QUANT_PATTERN.findall(t))

        row = {
            'filename': df['filename'].iloc[i],
            'core_langs': sorted(core),
            'frameworks_libs': sorted(fw),
            'cloud_devops': sorted(cloud),
            'databases': sorted(dbs),
            'action_verbs': verbs,
            'leadership': leaders,
            'quantified_claims': quantified,
            'top_tech_terms': top_terms,
            'tfidf_sum': tfidf_sum,
        }
        row['attention_score'] = round(attention_score(row), 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_sorted = out.sort_values(by=['attention_score','tfidf_sum'], ascending=[False, False])
    out_sorted.to_csv(RANKED_PATH, index=False)

    # Write concise report
    lines = []
    lines.append('Employer-Attention Ranking (Skills/Experience Keywords)\n')
    lines.append('=====================================================\n\n')
    for i, r in out_sorted.head(10).iterrows():
        lines.append(f"- {r['filename']}\n")
        lines.append(f"  score={r['attention_score']} | core_langs={', '.join(r['core_langs'])} | frameworks={', '.join(r['frameworks_libs'])}\n")
        lines.append(f"  cloud/devops={', '.join(r['cloud_devops'])} | db={', '.join(r['databases'])} | verbs={', '.join(r['action_verbs'])}\n")
        lines.append(f"  quantified_claims={r['quantified_claims']} | top_tech_terms={', '.join(r['top_tech_terms'])}\n\n")

    REPORT_PATH.write_text(''.join(lines), encoding='utf-8')

    print(f"Saved ranked CSV → {RANKED_PATH}")
    print(f"Saved report → {REPORT_PATH}")

if __name__ == '__main__':
    main()

