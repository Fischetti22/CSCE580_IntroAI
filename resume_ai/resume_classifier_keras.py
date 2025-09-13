#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Disable XLA entirely so no PTX/libdevice toolchain is required
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
# Force CPU execution to avoid CUDA/XLA toolchain issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- 1. Load resume texts ----------
# Disable TF JIT/XLA at framework level to reduce reliance on PTX toolchain
try:
    tf.config.optimizer.set_jit(False)
    tf.config.experimental.disable_mlir_bridge()
except Exception as e:
    print(f"Warning: could not fully disable TF JIT/MLIR: {e}")
df = pd.read_csv("out/resume_texts.csv")

if "text" not in df.columns:
    raise ValueError("resume_texts.csv must contain a 'text' column!")

# Check for label column (e.g., 'label' or 'Cluster')
if "label" in df.columns:
    y = df["label"].astype(str)  # job roles or folder labels
elif "KMeans_Cluster" in df.columns:
    y = df["KMeans_Cluster"].astype(str)  # cluster-based labels
else:
    raise ValueError("No labels found! Add 'label' or cluster column to resume_texts.csv.")

# Show label distribution
print("Label distribution (raw):")
print(y.value_counts())

# ---------- 2. Convert text to features ----------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X = tfidf.fit_transform(df["text"]).toarray()

# If only one unique label, fall back to unsupervised clusters so the classifier has >1 class
unique_labels = sorted(y.unique())
if len(unique_labels) < 2:
    n_samples = len(df)
    k = min(3, n_samples)  # try up to 3 clusters
    if k < 2:
        raise ValueError("Not enough samples to create clusters. Need at least 2 samples.")
    print(f"Only one label found ({unique_labels}). Falling back to KMeans clustering with initial k={k}.")
    # Try decreasing k until all clusters have at least 2 members, or k==2
    while k >= 2:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(X)
        counts = pd.Series(cluster_ids).value_counts()
        print(f"KMeans k={k} cluster sizes: {counts.to_dict()}")
        if counts.min() >= 2 or k == 2:
            y = pd.Series(cluster_ids, index=df.index).astype(str)
            break
        k -= 1

# Encode labels into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split (use stratify when possible), but avoid stratify if any class has <2 samples
class_counts = pd.Series(y_encoded).value_counts()
if class_counts.min() < 2:
    print("Warning: Some classes have <2 samples; proceeding without stratify for train/test split.")
    stratify = None
else:
    stratify = y_encoded

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=stratify
)

# ---------- 3. Define Keras Model ----------
num_classes = len(encoder.classes_)
print(f"num_classes = {num_classes}; classes = {list(encoder.classes_)}")

if num_classes == 2:
    inputs = keras.Input(shape=(X_train.shape[1],))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
else:
    inputs = keras.Input(shape=(X_train.shape[1],))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 

# ---------- 4. Train Model ----------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

# ---------- 5. Evaluate ----------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.3f}")

# ---------- 6. Save Model & Encoder ----------
model.save("resume_classifier.keras")

import joblib
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\n[Done] Saved model → resume_classifier.keras")
print("[Done] Saved TF-IDF vectorizer → tfidf_vectorizer.pkl")
print("[Done] Saved label encoder → label_encoder.pkl")

