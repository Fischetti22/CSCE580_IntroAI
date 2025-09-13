#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# ---------- Load Keyword Scores ----------
df = pd.read_csv("resume_keyword_scores.csv")

# Pick numeric features for clustering
feature_cols = ["good_hits", "bad_hits"]
if not all(c in df.columns for c in feature_cols):
    raise ValueError("resume_keyword_scores.csv must contain 'good_hits' and 'bad_hits' columns")

X = df[feature_cols]

# Scale features (important for fair clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- KMeans: Try Multiple k ----------
best_score = -1
best_k = None
best_labels = None

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    if len(set(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
    else:
        score = -1

    print(f"KMeans: k={k}, Silhouette Score={score:.3f}")

    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

print(f"\n[Best KMeans] k={best_k}, Silhouette Score={best_score:.3f}")

df["KMeans_Cluster"] = best_labels

# ---------- DBSCAN ----------
dbscan = DBSCAN(eps=0.5, min_samples=3)
labels_db = dbscan.fit_predict(X_scaled)

if len(set(labels_db)) > 1:
    db_score = silhouette_score(X_scaled, labels_db)
else:
    db_score = -1

print(f"[DBSCAN] Found {len(set(labels_db))} clusters, Silhouette Score={db_score:.3f}")

df["DBSCAN_Cluster"] = labels_db

# ---------- Save Results ----------
df.to_csv("resume_clusters_keywords.csv", index=False)
print("\n[Done] Saved clustering results → resume_clusters_keywords.csv")

