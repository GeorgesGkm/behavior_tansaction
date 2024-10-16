from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_loader import load_and_prepare_data
from model import predire_comportement_montant

app = FastAPI(
    debug=True,
    title="Transaction Behavior Prediction API", 
    description="An API for predicting whether a client's transaction amount is normal or anomalous using KMeans clustering models. This tool processes transaction data for different clients and returns insights based on machine learning predictions.",
    version="0.1",
)

data_file = 'data.csv' 
kmeans_models = load_and_prepare_data(data_file)

class TransactionRequest(BaseModel):
    client_id: str
    montant: float

@app.post("/predict")
async def predict_transaction(transaction: TransactionRequest):
    try:
        resultat = predire_comportement_montant(transaction.client_id, transaction.montant, kmeans_models)
        return {"client_id": transaction.client_id, "resultat": resultat}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route d'accueil
@app.get("/")
async def root():
    return {"message": "API de prédiction de comportement des transactions"}
