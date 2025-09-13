#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# ---------- Load Resume Texts ----------
df = pd.read_csv("resume_texts.csv")

if "text" not in df.columns:
    raise ValueError("resume_texts.csv must contain a 'text' column!")

# ---------- TF-IDF Features ----------
tfidf = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["text"])

# ---------- KMeans: Try Multiple k ----------
best_score = -1
best_k = None
best_labels = None

for k in range(2, 9):  # test k=2..8
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # Need at least 2 clusters
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels)
    else:
        score = -1

    print(f"KMeans: k={k}, Silhouette Score={score:.3f}")

    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

print(f"\n[Best KMeans] k={best_k}, Silhouette Score={best_score:.3f}")

# Add best KMeans labels to DataFrame
df["KMeans_Cluster"] = best_labels

# ---------- DBSCAN ----------
dbscan = DBSCAN(eps=0.5, min_samples=3)
labels_db = dbscan.fit_predict(X.toarray())

if len(set(labels_db)) > 1:
    db_score = silhouette_score(X, labels_db)
else:
    db_score = -1

print(f"[DBSCAN] Found {len(set(labels_db))} clusters, Silhouette Score={db_score:.3f}")

df["DBSCAN_Cluster"] = labels_db

# ---------- Save Results ----------
df.to_csv("resume_clusters.csv", index=False)
print("\n[Done] Saved clustering results → resume_clusters.csv")

