
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("legal_documents.csv", skipinitialspace=True)
data.columns = data.columns.str.strip()
data = data[data["category"].str.lower() != "category"]

X = data["text"]
y = data["category"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)
# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel training completed!")
