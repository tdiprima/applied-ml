"""
Reads support tickets and automatically sorts them into categories.
Uses TF-IDF (evaluate the importance of a word) + logistic regression.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load dataset
print("Loading support tickets dataset...")
df = pd.read_csv('support_tickets.csv')
print(f"Dataset: {len(df)} tickets, {df['label'].nunique()} categories")
print(df['label'].value_counts())

# Prepare data
X = df['text']
y = df['label']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Build simple pipeline: TF-IDF + Logistic Regression (deploy-ready)
print("\nTraining model...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
model = LogisticRegression(max_iter=1000, random_state=42)  # Converges fast

pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('logreg', model)
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.2%}")  # Expect ~85-90% on this data
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Sample predictions (new tickets)
print("\nSample predictions on new tickets:")
samples = [
    "App crashes on login iOS.",
    "Add dark mode please.",
    "Charged $20 twice.",
    "Reset password not working.",
    "How to export CSV?"
]
for text in samples:
    pred = pipeline.predict([text])[0]
    print(f"'{text}' -> {pred}")
