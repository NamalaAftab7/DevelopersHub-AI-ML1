# Import necessary libraries
from flask import Flask, request, render_template
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model and vectorizer from pickle files
model = pickle.load(open('sentiment_model.pkl', 'rb'))           # Logistic Regression model
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))     # TF-IDF vectorizer

# Define the home route to display the form
@app.route('/')
def home():
    # Render the HTML template (index.html) when user accesses the root URL
    return render_template('index.html')

# Define the prediction route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text entered by the user from the form
    review = request.form['review']
    
    # Vectorize the input review using the loaded TF-IDF vectorizer
    review_vec = vectorizer.transform([review])
    
    # Use the trained model to make a prediction (1 = Positive, 0 = Negative)
    prediction = model.predict(review_vec)[0]
    
    # Convert numeric prediction to text label
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    # Re-render the page and display the prediction result
    return render_template('index.html', prediction=sentiment)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)  # debug=True enables auto-reloading and error messages during development
