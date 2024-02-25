import pickle
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the pre-trained Random Forest Classifier model and scaler using pickle
with open('sleep_disorder_model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

class SleepData(BaseModel):
    quality_of_sleep: float = Field(..., ge=0.0, le=10.0)
    sleep_duration: float = Field(..., ge=0.0)
    stress_level: float = Field(..., ge=0.0, le=10.0)
    physical_activity_level: float = Field(..., ge=0.0, le=10.0)

@app.post("/predict")
def predict_sleep_disorder(data: SleepData):
    try:
        input_data = [[data.quality_of_sleep, data.sleep_duration, data.stress_level, data.physical_activity_level]]
        input_data = scaler.transform(input_data)
        prediction = model.predict_proba(input_data)

        if prediction is None or prediction[0] is None:
            raise HTTPException(status_code=500, detail="Unable to make a prediction for the given input.")

        result = False if prediction[0][1] > 0.5 else True

        return {"insomnia": result, "probability": float(prediction[0][1])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
