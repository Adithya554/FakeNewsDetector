import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

def load_dataset(file_path):
    """Loads dataset from a CSV file and removes missing values."""
    try:
        data = pd.read_csv(file_path)
        # Drop NaN values to prevent errors
        data.dropna(subset=['text', 'label'], inplace=True)
        return data
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs

def preprocess_text(text):
    """Tokenizes, removes punctuation, converts to lowercase, and stems words."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return " ".join(stemmed_tokens)

def train_model(X_train, y_train):
    """Trains a Naive Bayes classifier with TF-IDF vectorization."""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "fake_news_model.pkl")

    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on test data and prints accuracy & classification report."""
    y_pred = model.predict(X_test)
    print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
def predict(text):
    clean_text= preprocess_text(text)
    model = joblib.load("fake_news_model.pkl")
    prediction=model.predict(clean_text)
    return prediction

def main():
    """Main function to load data, preprocess, train, and evaluate model."""
    # Load dataset
    data = load_dataset("train.csv")

    # Prevent training on an empty dataset
    if data.empty:
        print("❌ No valid data found. Exiting.")
        return

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()


