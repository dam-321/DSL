# ==========================================
# NAIVE BAYES - SPAM DETECTION
# ==========================================

import pandas as pd
import zipfile, os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load ZIP
with zipfile.ZipFile("D:\Copy of sms+spam+collection (1).zip", 'r') as zip_ref:
    zip_ref.extractall("data")

df = pd.read_csv("data/SMSSpamCollection", sep='\t', names=["label","message"])

# Encode labels
df['label'] = df['label'].map({'ham':0, 'spam':1})

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))