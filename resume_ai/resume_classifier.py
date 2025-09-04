import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample resume dataset
data = {
    "text": [
        "Experienced in Python, ML, data analysis, statistics",
        "Skilled in Java, Spring Boot, microservices, APIs",
        "Expert in SQL, Tableau, data visualization, machine learning",
        "Developed REST APIs, worked on cloud deployments in AWS",
        "Research in NLP, transformers, and deep learning",
        "Built mobile apps using Android SDK, Kotlin, Firebase"
    ],
    "label": [
        "Data Scientist", "Software Engineer",
        "Data Scientist", "Software Engineer",
        "Data Scientist", "Software Engineer"
    ]
}

df = pd.DataFrame(data)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))

