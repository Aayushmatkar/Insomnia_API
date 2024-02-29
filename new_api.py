import numpy as np
from fastapi import FastAPI, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import pickle

app = FastAPI()

# Load the pre-trained Random Forest Classifier model and scaler
model = pickle.load(open('sleep_disorder_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Setup CORS
origins = [
    "http://localhost:3000",  # Update this with the actual origin of your frontend
    "https://insomnia2024-3f07450b7cfb.herokuapp.com/predict",  # Add additional origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SleepData(BaseModel):
    quality_of_sleep: float
    sleep_duration: float
    stress_level: float
    physical_activity_level: float

@app.get("/")
def foobar():
    return {
        "Server is up and Running"
    }

@app.post("/predict")
def predict_sleep_disorder(data: SleepData):
    '''
    Endpoint for predicting sleep disorder using the trained model.
    '''
    try:
        # Create a NumPy array from the input data
        input_data = np.array([[data.quality_of_sleep, data.sleep_duration, data.stress_level, data.physical_activity_level]])

        # Standardize the input data using the previously fitted scaler
        input_data = scaler.transform(input_data)

        # Make prediction using the pre-trained model
        prediction = model.predict_proba(input_data)

        # Check if prediction is not None
        if prediction is not None and prediction[0] is not None:
            # Determine result based on a different threshold (e.g., 0.7)
            result = False if prediction[0][1] > 0.5 else True

            # Return the prediction result as a boolean and the predicted probability
            return {
                "Insomnia": result,
                
            }
        else:
            # Handle the case where prediction is None
            return {"SleepDisorder": None, "error_message": "Unable to make a prediction for the given input."}

    except Exception as e:
        # Return an error message if an exception occurs
        return {"error_message": f"An error occurred: {str(e)}"}
