#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

REPORT_PATH = Path("out/analysis_report.txt")
MODEL_PATH = Path("out/resume_classifier_sklearn.joblib")
PIPELINE_PATH = Path("out/tfidf_kmeans_pipeline.joblib")
ASSIGNMENTS_PATH = Path("out/cluster_assignments.csv")
DATA_CSV = Path("out/resume_texts.csv")


def ensure_out_dir():
    Path("out").mkdir(parents=True, exist_ok=True)


def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " URL ", t)
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " EMAIL ", t)
    t = re.sub(r"\b\+?\d[\d\s().-]{6,}\b", " PHONE ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset CSV at {csv_path}")
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV must include a 'text' column")
    # Optional columns for reporting:
    if "filename" not in df.columns:
        df["filename"] = [f"doc_{i}" for i in range(len(df))]
    df["text"] = df["text"].astype(str).map(clean_text)
    return df


def supervised_pipeline(df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    X = df["text"].values
    y = df[label_col].astype(str).values
    X, y = shuffle(X, y, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_df": [0.9, 1.0],
        "tfidf__sublinear_tf": [True],
        "clf__C": [0.1, 1, 3, 10],
        "clf__solver": ["liblinear"],  # good for smaller feature spaces
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    # Save best model
    joblib.dump(grid.best_estimator_, MODEL_PATH)

    return {
        "mode": "supervised",
        "best_params": grid.best_params_,
        "cv_macro_f1": grid.best_score_,
        "test_report": report,
        "confusion_matrix": cm.tolist(),
    }


def choose_k_silhouette(X_tfidf: Any, k_min: int, k_max: int) -> int:
    best_k, best_score = None, -1
    for k in range(k_min, k_max + 1):
        if k <= 1:
            continue
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_tfidf)
            # Silhouette requires at least 2 labels and less than n_samples labels
            if len(set(labels)) <= 1 or len(set(labels)) >= len(labels):
                continue
            score = silhouette_score(X_tfidf, labels, metric="euclidean")
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k or 2


def top_terms_per_cluster(tfidf: TfidfVectorizer, X_tfidf, labels: np.ndarray, top_n: int = 12):
    feature_names = np.array(tfidf.get_feature_names_out())
    tops = {}
    for c in sorted(set(labels)):
        mask = labels == c
        if mask.sum() == 0:
            tops[c] = []
            continue
        # mean tfidf per feature within cluster
        mean_vec = X_tfidf[mask].mean(axis=0).A1
        idx = np.argsort(mean_vec)[::-1][:top_n]
        tops[c] = [(feature_names[i], float(mean_vec[i])) for i in idx]
    return tops


def clustering_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["text"].astype(str).tolist()
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(texts)

    n = len(texts)
    k_max = min(6, max(2, n - 1))
    k = choose_k_silhouette(X_tfidf, 2, k_max)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_tfidf)

    # Save pipeline for future use
    pipe = Pipeline([("tfidf", tfidf), ("kmeans", kmeans)])
    joblib.dump(pipe, PIPELINE_PATH)

    # Write cluster assignments with filenames
    df_out = df[["filename"]].copy()
    df_out["cluster"] = labels
    df_out.to_csv(ASSIGNMENTS_PATH, index=False)

    # Summaries
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    tops = top_terms_per_cluster(tfidf, X_tfidf, labels)

    # Examples per cluster (up to 5 filenames)
    examples = {}
    for c in sorted(set(labels)):
        examples[c] = df.loc[np.where(labels == c)[0], "filename"].head(5).tolist()

    return {
        "mode": "clustering",
        "k": int(k),
        "counts": {int(k_): int(v) for k_, v in counts.items()},
        "top_terms": {int(k_): v for k_, v in tops.items()},
        "examples": {int(k_): v for k_, v in examples.items()},
    }


def write_report(info: Dict[str, Any]):
    ensure_out_dir()
    lines = []
    lines.append("Resume Analysis Report\n")
    lines.append("======================\n\n")
    lines.append(f"Mode: {info['mode']}\n\n")

    if info["mode"] == "supervised":
        lines.append("Best Hyperparameters:\n")
        lines.append(json.dumps(info["best_params"], indent=2) + "\n\n")
        lines.append(f"CV macro-F1: {info['cv_macro_f1']:.4f}\n\n")
        lines.append("Test Classification Report:\n")
        lines.append(info["test_report"] + "\n")
        lines.append("Confusion Matrix:\n")
        lines.append(json.dumps(info["confusion_matrix"]) + "\n")
        lines.append(f"Saved model → {MODEL_PATH}\n")
    else:
        lines.append(f"Chosen k (clusters): {info['k']}\n\n")
        lines.append("Cluster sizes:\n")
        lines.append(json.dumps(info["counts"], indent=2) + "\n\n")
        lines.append("Top terms per cluster:\n")
        for c, terms in info["top_terms"].items():
            lines.append(f"- Cluster {c}: " + ", ".join(w for w, _ in terms) + "\n")
        lines.append("\nExample files per cluster (up to 5):\n")
        for c, files in info["examples"].items():
            lines.append(f"- Cluster {c}: " + ", ".join(files) + "\n")
        lines.append(f"\nSaved pipeline → {PIPELINE_PATH}\n")
        lines.append(f"Saved assignments → {ASSIGNMENTS_PATH}\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main():
    ensure_out_dir()
    df = load_data(DATA_CSV)

    # Decide supervised vs clustering
    label_col = None
    if "label" in df.columns and df["label"].astype(str).nunique() >= 2:
        label_col = "label"
    elif "KMeans_Cluster" in df.columns and df["KMeans_Cluster"].astype(str).nunique() >= 2:
        label_col = "KMeans_Cluster"

    if label_col:
        info = supervised_pipeline(df, label_col)
    else:
        info = clustering_pipeline(df)

    write_report(info)

    # Console summary
    print(f"Mode: {info['mode']}")
    if info["mode"] == "supervised":
        print("Best params:", info["best_params"]) 
        print(f"CV macro-F1: {info['cv_macro_f1']:.4f}")
        print(f"Saved model → {MODEL_PATH}")
        print(f"Report → {REPORT_PATH}")
    else:
        print(f"k = {info['k']} | sizes: {info['counts']}")
        print(f"Saved pipeline → {PIPELINE_PATH}")
        print(f"Assignments → {ASSIGNMENTS_PATH}")
        print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()

