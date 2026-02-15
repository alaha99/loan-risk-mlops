from fastapi import FastAPI
from inference.predict import predict

app = FastAPI()

@app.get("/predict/{applicant_id}")
def get_prediction(applicant_id: str):
    return predict(applicant_id)
