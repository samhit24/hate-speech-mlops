import joblib
import re
from nltk.corpus import stopwords
import uvicorn
import nltk
from fastapi import FastAPI
from pydantic import BaseModel

# Download stopwords is now handled in the Dockerfile
# Define the set of English stopwords
stopwords_set = set(stopwords.words('english'))

# Create the FastAPI app instance
app = FastAPI()

# Load the trained model and vectorizer
try:
    model = joblib.load("models/resampled_logistic_regression.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
except FileNotFoundError:
    raise RuntimeError("Model or vectorizer files not found. Please run the notebook to train and save them.")

# Define the input and output schemas
class Tweet(BaseModel):
    text: str

# Define the cleaning function (copied from the notebook)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+|https?://\S+|www\.\S+|\W+|\d+', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text_words = text.split()
    text = ' '.join([word for word in text_words if word not in stopwords_set])
    return text

# Define the prediction function
def predict_class(text: str):
    cleaned_text = clean_text(text)
    # The vectorizer needs a list of strings
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return int(prediction[0])

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Hate Speech Detection API!"}

# Define the health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Define the prediction endpoint
@app.post("/predict")
def predict(tweet: Tweet):
    prediction = predict_class(tweet.text)
    # Map the prediction number to a label
    labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
    predicted_label = labels[prediction]
    return {"original_tweet": tweet.text, "predicted_class": prediction, "predicted_label": predicted_label}

# The main block to run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)