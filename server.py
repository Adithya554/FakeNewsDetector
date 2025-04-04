from flask import Flask, request, jsonify
import joblib
import analysis
from analysis import preprocess_text  # Import the preprocessing function

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("fake_news_model.pkl")  # Load the model only once
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None  # Set to None if the model can't be loaded

@app.route("/predict", methods=["POST"])
def predict():
    """Receives JSON input, processes it, and returns a fake/real news prediction."""
    
    if model is None:
        return jsonify({"error": "Model is not loaded properly."}), 500

    try:
        data = request.get_json()

        # Check if 'text' field is present in the request
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field in the request"}), 400

        text = data["text"]

        # Preprocess the text before prediction
        processed_text = preprocess_text(text)

        # Predict on preprocessed text
        prediction = model.predict([processed_text])[0]  
        result = "Fake News" if prediction == 1 else "Real News"
        
        return jsonify({"prediction": result})
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log the error to the console
        return jsonify({"error": f"Error during prediction: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
