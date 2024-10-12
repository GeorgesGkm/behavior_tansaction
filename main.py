from fastapi import FastAPI
from pydantic import BaseModel
from data_loader import load_data
from model import predict_amount

scaler, kmeans = load_data('data.csv')

app = FastAPI()

class AmountInput(BaseModel):
    amount: float

@app.post("/predict")
def predict_comportement(input_data: AmountInput):
    amount = input_data.amount
    resultat = predict_amount(amount, scaler, kmeans)
    return {"montant": amount, "prediction": resultat}
