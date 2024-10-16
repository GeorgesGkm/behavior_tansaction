import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_and_prepare_data(file_path: str):
    data = pd.read_csv(file_path)
    
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    if 'clientId' not in data.columns or 'montant' not in data.columns:
        raise ValueError("Les colonnes 'clientId' et 'montant' doivent être présentes dans les données.")
    
    montant_data = data[['clientId', 'montant']].copy()
    
    scaler = StandardScaler()
    kmeans_models = {}

    for client_id, group in montant_data.groupby('clientId'):
        if group['montant'].nunique() > 1:  # Vérifie qu'il y a plus d'un montant unique
            montant_data_scaled = scaler.fit_transform(group[['montant']])

            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(montant_data_scaled)

            kmeans_models[client_id] = (kmeans, scaler)
        else:
            print(f"Client {client_id} a moins de données uniques, pas de modèle créé.")
    
    return kmeans_models
