# 📰 Fake News Detector  

This project is a *Machine Learning-powered web application* that **detects fake news** based on the article's content. It uses **Natural Language Processing (NLP) techniques** and a **Random Forest Classifier** to classify news as **fake or real**. The web application is built using **Flask** and deployed on **Render**.

---

## **📂 Project Structure**  

```
📁 FakeNewsDetector
├── 📄 model_training.py      # Data preprocessing and model training script
├── 📄 app.py                 # Flask web application
├── 📄 requirements.txt       # Dependencies for the project
├── 📄 train.csv              # Fake news dataset
├── 📁 templates/             # HTML templates for the web interface
│   ├── index.html            # Homepage for news input
│   ├── result.html           # Displays the prediction result
└── 📄 README.md              # Project documentation
```

---

## **🗂 Dataset**  

The model is trained on the **Fake News Dataset** from Kaggle:  
🔗 [Fake News Competition Dataset](https://www.kaggle.com/c/fake-news/data)

This dataset contains news articles labeled as **fake** or **real**, with the following columns:  

- `id` → Unique identifier  
- `title` → Headline of the article  
- `author` → Author of the article  
- `text` → Main news content  
- `label` → **0 (Fake News)** or **1 (Real News)**  

---

# **🚀 Pipeline Overview**  

The **Fake News Detector** consists of the following steps:

### **1️⃣ Data Preprocessing & Cleaning**  
- **Remove punctuation and special characters** from text.  
- **Convert text to lowercase** for uniformity.  
- **Tokenization**: Splitting text into individual words.  
- **Remove stopwords** like "is", "the", "and", etc., which do not contribute to meaning.  

#### **Preprocessing Code (model_training.py)**  
```python
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    words = [word for word in words if word.isalpha() and word not in stop_words]  
    return " ".join(words)
```

---

### **2️⃣ Feature Engineering: TF-IDF Vectorization**  
- Converts text into **numerical vectors** using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- This helps the machine learning model understand the importance of words in news articles.  
- Limits to **5000 most important words** to avoid overfitting.  

#### **TF-IDF Vectorization Code**  
```python
vectorizer = TfidfVectorizer(max_features=5000)  # Extract most relevant words
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

---

### **3️⃣ Model Training: Random Forest Classifier**  
- The **Random Forest Classifier** is used for classification.  
- It is an ensemble learning method that uses multiple decision trees.  
- The model is trained on **80%** of the dataset and tested on the remaining **20%**.

#### **Model Training Code (model_training.py)**  
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)
```

---

### **4️⃣ Model Evaluation**  
- The model is tested on **unseen data**.  
- Accuracy and other performance metrics are calculated.  

#### **Evaluation Code**  
```python
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

### **5️⃣ Saving the Model for Deployment**  
- The trained model and vectorizer are saved using `joblib`.  
- These files are used later in the **Flask web application**.

#### **Saving the Model**  
```python
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
```

---

# **🌐 Flask Web Application**  

### **🔹 app.py (Flask Backend)**
This script loads the **trained model** and **vectorizer** to classify user input news.

### **1️⃣ Downloading Model from Google Drive**  
Since deployment on Render requires persistent storage, the trained model is hosted on **Google Drive** and downloaded automatically.  

#### **Downloading Model in app.py**  
```python
import gdown

model_url = "https://drive.google.com/uc?id=1rEjXwfeqtiMzouFCvgJYAWrbHOMc9itY"
vectorizer_url = "https://drive.google.com/uc?id=18Jkt58N6_t-DM6zFuGcT2CPRivch2Wvt"

if not os.path.exists("model.pkl"):
    gdown.download(model_url, "model.pkl", quiet=False)

if not os.path.exists("vectorizer.pkl"):
    gdown.download(vectorizer_url, "vectorizer.pkl", quiet=False)
```

---

### **2️⃣ Web Routes**  

| Route         | Functionality |
|--------------|--------------|
| `/`          | Loads the homepage where users can enter news content. |
| `/predict`   | Takes user input, processes it, and returns whether the news is **Fake or Real**. |

#### **Flask Routes**  
```python
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
```

---

# **📌 How to Run the Project Locally**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Adithya554/FakeNewsDetector.git
cd FakeNewsDetector
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Flask App**  
```bash
python app.py
```

### **4️⃣ Open the Web App**  
Go to **`http://127.0.0.1:5000/`** in your browser.

---

# **🌍 Deployment on Render**
This project is **deployed on Render**, a cloud platform for web applications.  

https://fakenewsdetector-4.onrender.com
