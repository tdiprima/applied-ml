#!/usr/bin/env python3
"""
Classifies emails as spam or not using Naive Bayes.
Trains in milliseconds, prints accuracy and speed benchmarks.
"""

import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
print("📥 Loading dataset...")
df = pd.read_csv('data/emails.csv')
print(f"Dataset: {len(df)} emails ({df['label'].value_counts().to_dict()})")

# Prep data: Vectorize text (bag-of-words, perfect for NB speed)
vectorizer = CountVectorizer(stop_words='english', max_features=5000)  # Limits vocab for speed/deployment
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split: 80/20, stratified for balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train NB model
print("\n🚀 Training Naive Bayes...")
start_time = time.time()
model = MultinomialNB()  # Default alpha=1.0 for speed/robustness
model.fit(X_train, y_train)
train_time = (time.time() - start_time) * 1000  # ms
print(f"✅ Training time: {train_time:.2f} ms (ultra-fast!)")

# Predict on test set
print("\n⚡ Testing inference speed...")
start_time = time.time()
y_pred = model.predict(X_test)
infer_time = (time.time() - start_time) * 1000  # ms total
infer_per_email = infer_time / X_test.shape[0]  # ms/email
print(f"✅ Inference time: {infer_time:.2f} ms total ({infer_per_email:.3f} ms/email)")

# Accuracy (good enough for filtering; speed wins)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"✅ Accuracy: {accuracy:.1f}%")

# Example predictions (understandable output)
test_samples = ['Meeting tomorrow, see attached.',  # Ham-like
                'WIN $1M NOW!!! CLICK http://scam.link']  # Spam-like
sample_vec = vectorizer.transform(test_samples)
preds = model.predict(sample_vec)
probs = model.predict_proba(sample_vec)
print("\n🔍 Sample predictions:")
for i, (text, pred, prob) in enumerate(zip(test_samples, preds, probs)):
    print(f"Email: '{text[:50]}...'\nPred: {pred} (prob: {prob[1]:.2f} spam)\n")

print("\n🎉 Ready for deployment! Pipe emails to classify in prod.")
