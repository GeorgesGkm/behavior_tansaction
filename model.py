import numpy as np

def predire_comportement_montant(client_id: str, nouveau_montant: float, kmeans_models: dict):
    if client_id not in kmeans_models:
        return "Aucun modèle disponible pour ce client."

    kmeans, scaler = kmeans_models[client_id]

    nouvelle_donnee = np.array([[nouveau_montant]])

    nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)

    cluster = kmeans.predict(nouvelle_donnee_scaled)

    if cluster[0] == 0:
        return "Comportement normal."
    else:
        return "Comportement anormal."
