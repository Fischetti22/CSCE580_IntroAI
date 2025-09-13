#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import normalize

BASE = Path('/home/droski/Desktop/School/Fall25/AI/resume_ai/out')
DATA_CSV = BASE / 'resume_texts.csv'
PIPELINE_PATH = BASE / 'tfidf_kmeans_pipeline.joblib'
RANKED_PATH = BASE / 'ranked_resumes.csv'
CLUSTER_LABELS_JSON = BASE / 'cluster_label_map.json'


def top_terms_for_vector(tfidf, row_vec, top_n=8):
    feature_names = np.array(tfidf.get_feature_names_out())
    if hasattr(row_vec, 'toarray'):
        row = row_vec.toarray()[0]
    else:
        row = np.asarray(row_vec)
    idx = np.argsort(row)[::-1][:top_n]
    return [str(feature_names[i]) for i in idx if row[i] > 0]


def top_terms_per_cluster(tfidf, X, labels, top_n=8):
    terms = {}
    feature_names = np.array(tfidf.get_feature_names_out())
    # mean tfidf per cluster
    for c in sorted(set(labels)):
        mask_idx = np.where(labels == c)[0]
        if len(mask_idx) == 0:
            terms[c] = []
            continue
        mean_vec = X[mask_idx].mean(axis=0).A1
        idx = np.argsort(mean_vec)[::-1][:top_n]
        terms[c] = [str(feature_names[i]) for i in idx if mean_vec[i] > 0]
    return terms


def cosine_sim_matrix(A, B):
    A_n = normalize(A)
    B_n = normalize(B)
    return A_n @ B_n.T


def main():
    if not DATA_CSV.exists():
        raise SystemExit(f'Missing {DATA_CSV}')
    if not PIPELINE_PATH.exists():
        raise SystemExit(f'Missing {PIPELINE_PATH}. Run resume_analysis_sklearn.py first.')

    df = pd.read_csv(DATA_CSV)
    pipe = joblib.load(PIPELINE_PATH)  # Pipeline(tfidf, kmeans)
    tfidf = pipe.named_steps['tfidf']
    kmeans = pipe.named_steps['kmeans']

    X = tfidf.transform(df['text'].astype(str).tolist())
    labels = kmeans.predict(X)

    # Cosine similarity to cluster centers
    centers = kmeans.cluster_centers_  # in tf-idf feature space
    sims = cosine_sim_matrix(X, centers)  # shape: (n_samples, k)
    chosen_sim = sims[np.arange(X.shape[0]), labels]

    # z-score within cluster
    zscores = np.zeros_like(chosen_sim)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        vals = chosen_sim[idx]
        mu = vals.mean()
        sd = vals.std() if vals.std() > 1e-8 else 1.0
        zscores[idx] = (vals - mu) / sd

    # Friendly labels from top terms per cluster
    cluster_terms = top_terms_per_cluster(tfidf, X, labels, top_n=6)
    cluster_label_map = {int(c): ' / '.join(cluster_terms[c][:3]).title() if cluster_terms[c] else f'Cluster {c}'
                         for c in cluster_terms}

    # Per-row top terms
    per_row_terms = [', '.join(top_terms_for_vector(tfidf, X[i], top_n=6)) for i in range(X.shape[0])]

    out = df.copy()
    out['KMeans_Cluster'] = labels
    out['Cluster_Label'] = [cluster_label_map[int(c)] for c in labels]
    out['Cluster_Cosine'] = chosen_sim
    out['Cluster_ZScore'] = zscores
    out['Top_Terms'] = per_row_terms

    # Sort by z-score desc (best within their cluster)
    out_sorted = out.sort_values(by=['Cluster_ZScore', 'Cluster_Cosine'], ascending=[False, False])
    out_sorted.to_csv(RANKED_PATH, index=False)

    # Save cluster label mapping
    with open(CLUSTER_LABELS_JSON, 'w') as f:
        json.dump(cluster_label_map, f, indent=2)

    # Print top 5
    top5 = out_sorted[['filename', 'KMeans_Cluster', 'Cluster_Label', 'Cluster_Cosine', 'Cluster_ZScore']].head(5)
    print('Top 5 resumes by within-cluster similarity:')
    for _, row in top5.iterrows():
        print(f"- {row['filename']} | cluster={int(row['KMeans_Cluster'])} ({row['Cluster_Label']}) | sim={row['Cluster_Cosine']:.3f} | z={row['Cluster_ZScore']:.2f}")

    print(f"\nSaved ranked list → {RANKED_PATH}")
    print(f"Saved cluster label map → {CLUSTER_LABELS_JSON}")


if __name__ == '__main__':
    main()

