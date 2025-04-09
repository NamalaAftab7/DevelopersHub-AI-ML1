from flask import Flask, request, render_template
import pickle
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK Data (only needed once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Setup Flask App
app = Flask(__name__)

# Load Trained Model and Vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing Function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    processed = preprocess_text(news)
    transformed = vectorizer.transform([processed]).toarray()
    
    prediction = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed)[0][prediction] * 100  # confidence in %

    result = "Fake News ❌" if prediction == 1 else "Real News ✅"
    confidence_str = f"{confidence:.2f}% confident"
    
    return render_template("index.html", prediction=result, confidence=confidence_str)

# Run the App
if __name__ == "__main__":
    app.run(debug=True)
