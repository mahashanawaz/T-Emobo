#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the dataset
filename = os.path.join(os.getcwd(), "feedback_labeled.csv")
df = pd.read_csv(filename)

print("Sample data:")
print(df.head())
print("Columns:", df.columns)


# Define features and labels
X = df['FeedbackText']
y = df['Sentiment']

# Train-test split (75% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1234, stratify=y)

print("Train label counts:\n", y_train.value_counts())
print("Test label counts:\n", y_test.value_counts())

# Create and fit TF-IDF vectorizer on training data
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train)

# Transform train and test data
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Predict probabilities and labels on test set
probability_predictions = model.predict_proba(X_test_tfidf)[:, 1]
class_label_predictions = model.predict(X_test_tfidf)

# Compute and print AUC score
auc = roc_auc_score(y_test, probability_predictions)
print(f'AUC on the test data: {auc:.4f}')

# Show size of feature space and a few features
print(f'Feature space size: {len(tfidf_vectorizer.vocabulary_)}')
print('Sample features:', list(tfidf_vectorizer.vocabulary_.items())[:5])

# Example review checks
print('\nExample review predictions:\n')

for idx in [0, 1]:  # just first two test samples for demo
    print(f"Review #{idx+1}: {X_test.iloc[idx]}")
    print(f"Predicted sentiment (1=positive): {class_label_predictions[idx]}")
    print(f"Actual sentiment: {y_test.iloc[idx]}\n")


# Function to predict sentiment on new text inputs
def predict_sentiment(text):
    vectorized_text = tfidf_vectorizer.transform([text])
    return model.predict_proba(vectorized_text)[0][1]

# Test new sample texts
print("Sample prediction - Positive:", predict_sentiment("This service is great!"))
print("Sample prediction - Negative:", predict_sentiment("Terrible customer support."))
