from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the model and preprocessing objects
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Define a Pydantic model for the request body
class Status(BaseModel):
    externalStatus: str

# Initialize the FastAPI app
app = FastAPI()

# Define the predict function
@app.post("/predict/")
async def predict(status: Status):
    # Preprocess the input
    processed_input = vectorizer.transform([status.externalStatus])

    # Make prediction
    prediction = model.predict(processed_input)

    # Postprocess the prediction
    processed_output = encoder.inverse_transform(prediction)

    return {"internalStatus": processed_output[0]}
