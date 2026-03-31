import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

print("Loading dataset...")
df = pd.read_csv("data/clickbait_data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df['headline'], df['clickbait'], test_size=0.2, random_state=42
)

model = make_pipeline(
    TfidfVectorizer(stop_words='english', lowercase=True, max_features=5000),
    MultinomialNB()
)

print("Training model...")
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Training Complete! Accuracy: {accuracy * 100:.2f}%")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/clickbait_model.pkl")
print("Model saved to models/clickbait_model.pkl")