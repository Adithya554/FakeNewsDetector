import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("train.csv", low_memory=False)  # Replace with your Kaggle dataset filename

df.dropna(subset=["label"], inplace=True)

# Ensure your dataset has 'text' (news content) and 'label' (0 for fake, 1 for real)
stop_words=set(stopwords.words('english'))
# Text Preprocessing Function
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    words = [word for word in words if word.isalpha() and word not in stop_words]  
    return " ".join(words)

df["text"] = df["text"].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to reduce overfitting
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
