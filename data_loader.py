import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(fichier_csv):
    data = pd.read_csv('data.csv')
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    montant_data = data[['montant']].copy()

    scaler = StandardScaler()
    montant_data_scaled = scaler.fit_transform(montant_data)

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(montant_data_scaled)

    return scaler, kmeans
