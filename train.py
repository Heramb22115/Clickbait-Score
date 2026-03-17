import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

print("Loading the 32,000 headline dataset...")
# Load the CSV file from your data folder
df = pd.read_csv("data/clickbait_data.csv")

# The CSV has two columns: 'headline' (the text) and 'clickbait' (1 or 0)
# We split 80% of it to train the brain, and hold back 20% to test how smart it got
X_train, X_test, y_train, y_test = train_test_split(
    df['headline'], df['clickbait'], test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} headlines. Testing on {len(X_test)} unseen headlines...")

# Build the pipeline
model = make_pipeline(
    TfidfVectorizer(stop_words='english', lowercase=True, max_features=5000),
    MultinomialNB()
)

# Train the model
model.fit(X_train, y_train)

# Give it an exam using the 20% of data it hasn't seen yet
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Training Complete! Real-World Accuracy: {accuracy * 100:.2f}%")

# Save the supercharged model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/clickbait_model.pkl")
print("New model saved successfully!")