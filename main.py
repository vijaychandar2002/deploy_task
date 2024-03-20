import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import requests
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Preprocess the data
def preprocess_data(df):
    # Convert the 'externalStatus' column to a TF-IDF vector
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['externalStatus'])

    # Encode the 'internalStatus' column
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['internalStatus'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer, encoder

# Create the Random Forest model
def create_rf_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

# Train the model
def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Load the data
url = "https://gist.githubusercontent.com/farhaan-settyl/ecf9c1e7ab7374f18e4400b7a3d2a161/raw/f94652f217eeca83e36dab9d08727caf79ebdecf/dataset.json"
response = requests.get(url)
data = response.json()
df = pd.json_normalize(data)

# Preprocess the data
X_train, X_test, y_train, y_test, vectorizer, encoder = preprocess_data(df)

# Create the Random Forest model
model = create_rf_model()

# Train the model
model = train_model(X_train, y_train, model)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Save the model and preprocessing objects
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

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
