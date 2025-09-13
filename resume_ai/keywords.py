#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, sys, json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd

# text extraction
import pdfplumber
from docx import Document

# features & similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# File readers
# -----------------------------
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def try_textract(path: Path) -> str:
    try:
        import textract
        return textract.process(str(path)).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def read_any(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".txt":
            return read_txt(path)
        elif ext == ".docx":
            return read_docx(path)
        elif ext == ".pdf":
            return read_pdf(path)
        else:
            # last-resort if you installed textract
            return try_textract(path)
    except Exception:
        # fallback if primary reader failed
        return try_textract(path)


# -----------------------------
# Load resumes from folder
#   label = parent directory name (useful but optional)
# -----------------------------
def load_resumes(root: Path) -> pd.DataFrame:
    rows = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() in [".pdf", ".docx", ".txt"] or True:
            text = read_any(f)
            text = (text or "").strip()
            if not text:
                continue
            label = f.parent.name  # folder name as label (can be a role)
            rows.append({"filename": str(f.resolve()), "label": label, "text": text})
    df = pd.DataFrame(rows)
    if df.empty:
        print("[!] No readable files found. Check your folder and file types.", file=sys.stderr)
    return df


# -----------------------------
# Make word-count matrix (common words)
# -----------------------------
def build_bow(df_texts: pd.DataFrame,
              stopwords: str = "english",
              max_features: int = 5000,
              ngrams: Tuple[int, int] = (1, 2)) -> Tuple[pd.DataFrame, CountVectorizer]:
    vect = CountVectorizer(stop_words=stopwords, max_features=max_features, ngram_range=ngrams)
    X = vect.fit_transform(df_texts["text"])
    bow = pd.DataFrame(X.toarray(), index=df_texts["filename"], columns=vect.get_feature_names_out())
    return bow, vect


# -----------------------------
# Top common words overall & by label
# -----------------------------
def summarize_common_words(bow: pd.DataFrame, df_texts: pd.DataFrame, top_k: int = 30) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    overall_counts = bow.sum(axis=0).sort_values(ascending=False)
    overall_top = overall_counts.head(top_k).reset_index()
    overall_top.columns = ["token", "count"]

    by_label = {}
    for label, idx in df_texts.groupby("label").indices.items():
        sub = bow.iloc[idx]
        counts = sub.sum(axis=0).sort_values(ascending=False).head(top_k).reset_index()
        counts.columns = ["token", "count"]
        by_label[label] = counts
    return overall_top, by_label


# -----------------------------
# Keywords CSV helpers
#   We normalize messy headers & find good/bad columns
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace("-", "_")
                  .str.replace(r"[^0-9A-Za-z_]", "", regex=True)
    )
    return df

def detect_keyword_columns(cols: List[str]) -> Tuple[str, str]:
    # try to find columns containing "good" and "bad"
    lower = [c.lower() for c in cols]
    good_candidates = [c for c in cols if "good" in c.lower()]
    bad_candidates  = [c for c in cols if "bad"  in c.lower()]

    good_col = good_candidates[0] if good_candidates else None
    bad_col  = bad_candidates[0]  if bad_candidates  else None
    return good_col, bad_col

def split_terms(cell: str) -> List[str]:
    if pd.isna(cell): 
        return []
    # split on commas; strip whitespace; drop empties
    return [t.strip().lower() for t in str(cell).split(",") if t.strip()]


# -----------------------------
# Score each resume by keyword hits
# -----------------------------
def compile_matcher(words: List[str]) -> re.Pattern:
    # word-boundary match for each term; escape special chars
    words = [w for w in words if w]
    if not words:
        return re.compile(r"$a")  # matches nothing
    pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)

def score_with_keywords(df_texts: pd.DataFrame, kw_df: pd.DataFrame) -> pd.DataFrame:
    kw_df = normalize_columns(kw_df)
    good_col, bad_col = detect_keyword_columns(list(kw_df.columns))
    if not good_col or not bad_col:
        raise ValueError(
            f"Could not find keyword columns (good/bad). Got columns: {kw_df.columns.tolist()}\n"
            f"Expected something containing 'good' and 'bad' in the header names."
        )

    # build master sets
    all_good = set()
    all_bad = set()
    for _, row in kw_df.iterrows():
        all_good.update(split_terms(row[good_col]))
        all_bad.update(split_terms(row[bad_col]))

    good_pat = compile_matcher(sorted(all_good))
    bad_pat  = compile_matcher(sorted(all_bad))

    rows = []
    for _, r in df_texts.iterrows():
        text = r["text"]
        good_hits = good_pat.findall(text)
        bad_hits  = bad_pat.findall(text)

        rows.append({
            "filename": r["filename"],
            "label": r["label"],
            "good_hits": len(good_hits),
            "bad_hits": len(bad_hits),
            "good_terms_found": ", ".join(sorted(set(g.lower() for g in good_hits))),
            "bad_terms_found": ", ".join(sorted(set(b.lower() for b in bad_hits)))
        })

    return pd.DataFrame(rows)


# -----------------------------
# Default keywords 
# -----------------------------
DEFAULT_KEYWORDS = [
    {
        "Category": "Programming",
        "Good_Keywords": "python, java, c++, sql, r, javascript, scala",
        "Bad_Keywords": "basic computer skills, ms word, excel only"
    },
    {
        "Category": "Data Science",
        "Good_Keywords": "machine learning, deep learning, nlp, ai, pandas, tensorflow, pytorch, statistics",
        "Bad_Keywords": "guessing, approximate knowledge, some experience"
    },
    {
        "Category": "Cloud/DevOps",
        "Good_Keywords": "aws, azure, gcp, docker, kubernetes, ci/cd, linux",
        "Bad_Keywords": "familiar with cloud, tinkered, beginner"
    },
    {
        "Category": "Tools",
        "Good_Keywords": "git, jira, confluence, tableau, powerbi, spark",
        "Bad_Keywords": "microsoft paint, office only"
    },
    {
        "Category": "Soft Skills",
        "Good_Keywords": "leadership, teamwork, communication, adaptability, problem solving",
        "Bad_Keywords": "hardworking, go-getter, fast learner"
    },
    {
        "Category": "Certifications",
        "Good_Keywords": "aws certified, gcp certified, pmp, cfa, scrum master",
        "Bad_Keywords": "certificate of participation, attended workshop"
    },
    {
        "Category": "Experience Level",
        "Good_Keywords": "senior, lead, manager, 5+ years, expert",
        "Bad_Keywords": "entry-level, beginner, novice"
    },
    {
        "Category": "Impact Words",
        "Good_Keywords": "delivered, achieved, designed, implemented, optimized",
        "Bad_Keywords": "helped with, tried, attempted, exposed to"
    }
]

def ensure_keywords_csv(path: Path, force_overwrite: bool = False):
    if path.exists() and not force_overwrite:
        return
    df = pd.DataFrame(DEFAULT_KEYWORDS)
    df.to_csv(path, index=False)
    print(f"[+] Wrote default keyword dictionary → {path}")


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="PDF→text, word counts, and keyword scoring for local resumes."
    )
    ap.add_argument("--root", required=True, help="Root folder with resumes (PDF/DOCX/TXT).")
    ap.add_argument("--outdir", default="out", help="Where to write CSV outputs.")
    ap.add_argument("--keywords", default="resume_keywords.csv",
                    help="CSV with columns like Good_Keywords/Bad_Keywords. If missing, a default will be created.")
    ap.add_argument("--overwrite-default-keywords", action="store_true",
                    help="Overwrite the keywords CSV with the default set.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    kw_path = Path(args.keywords).resolve()

    # 1) Load resumes & extract text
    print(f"[+] Scanning resumes under: {root}")
    df_texts = load_resumes(root)
    if df_texts.empty:
        sys.exit(1)

    texts_csv = outdir / "resume_texts.csv"
    df_texts.to_csv(texts_csv, index=False)
    print(f"[+] Saved extracted texts → {texts_csv}")

    # 2) Build word-count matrix (common words / ngrams)
    bow, vect = build_bow(df_texts)
    bow_csv = outdir / "resume_word_counts.csv"
    bow.to_csv(bow_csv)
    print(f"[+] Saved bag-of-words matrix → {bow_csv}")

    # 3) Summaries of common words overall/by-label
    overall_top, by_label = summarize_common_words(bow, df_texts, top_k=40)
    overall_csv = outdir / "top_words_overall.csv"
    overall_top.to_csv(overall_csv, index=False)
    print(f"[+] Saved top overall words → {overall_csv}")
    for label, df_top in by_label.items():
        p = outdir / f"top_words_{re.sub(r'[^0-9A-Za-z]+','_',label)}.csv"
        df_top.to_csv(p, index=False)
        print(f"[+] Saved top words for label '{label}' → {p}")

    # 4) Cosine similarity between resumes (who's similar to whom?)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
    X = tfidf.fit_transform(df_texts["text"])
    sim = cosine_similarity(X)
    sim_df = pd.DataFrame(sim, index=df_texts["filename"], columns=df_texts["filename"])
    sim_csv = outdir / "resume_similarity_cosine.csv"
    sim_df.to_csv(sim_csv)
    print(f"[+] Saved cosine similarity matrix → {sim_csv}")

    # 5) Keyword scoring (create default keywords CSV if missing)
    ensure_keywords_csv(kw_path, force_overwrite=args.overwrite_default_keywords)
    kw_df = pd.read_csv(kw_path)
    scored = score_with_keywords(df_texts, kw_df)
    scored_csv = outdir / "resume_keyword_scores.csv"
    scored.to_csv(scored_csv, index=False)
    print(f"[+] Saved keyword hit scores → {scored_csv}")

    print("\n[Done] Open the CSVs in Excel:")
    print(f" - {texts_csv.name}: raw extracted text")
    print(f" - {bow_csv.name}: word/phrase counts per resume")
    print(f" - {overall_csv.name} & top_words_*.csv: frequent tokens")
    print(f" - {sim_csv.name}: pairwise similarity")
    print(f" - {scored_csv.name}: good/bad keyword hits per resume")


if __name__ == "__main__":
    main()

