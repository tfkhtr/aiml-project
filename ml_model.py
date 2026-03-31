# ================== ML MODEL ==================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

DATA_FILE = "training_data.txt"

# ------------------ LOAD DATA ------------------

def load_data():
    data = []

    if not os.path.exists(DATA_FILE):
        print("ERROR: training_data.txt not found!")
        exit()

    with open(DATA_FILE, "r") as f:
        for line in f:
            if "," in line:
                text, label = line.strip().split(",", 1)
                data.append((text.lower(), label.lower()))

    if not data:
        print("ERROR: training_data.txt is empty!")
        exit()

    return data

# ------------------ TRAIN ------------------

def train_model():
    data = load_data()

    texts = [x[0] for x in data]
    labels = [x[1] for x in data]

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    return model, vectorizer

# ------------------ PREDICT WITH CONFIDENCE ------------------

def predict_with_confidence(symptom, model, vectorizer):
    symptom = symptom.lower()
    X = vectorizer.transform([symptom])

    probs = model.predict_proba(X)[0]
    best_index = probs.argmax()

    predicted_label = model.classes_[best_index]
    confidence = probs[best_index]

    return predicted_label, confidence

# ------------------ SAVE LEARNING ------------------

def save_new_data(symptom, label):
    with open(DATA_FILE, "a") as f:
        f.write(f"{symptom.lower()},{label.lower()}\n")