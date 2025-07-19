# PRODIGY_AIML_04
Hand Gesture Recognition Model
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset (Assume labels and images are pre-processed)
# X: image features (e.g., flatten images or extracted features), y: gesture labels

# For example only: dummy dataset
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target  # Simulating gesture classes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
