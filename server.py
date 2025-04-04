from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import gdown
import pickle

# Initialize Flask app
app = Flask("app")

model_url="https://drive.google.com/drive/folders/1cVkckA7MB3EWLSiS_Liooa1Pqdy-lkHg?usp=drive_link"
model_path="model.pkl"
# Load the trained model and vectorizer

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

with open(model_path, "rb") as f:
    vectorizer, model = pickle.load(f)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha() and word not in stopwords.words("english")]
    return " ".join(words)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data.get("text", "")
        if not news_text.strip():
            return jsonify({"error": "No text provided"}), 400
        
        # Preprocess and vectorize the input text
        processed_text = preprocess_text(news_text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        result = "Real News" if prediction == 1 else "Fake News"
        
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
