import numpy as np

def predict_amount(amounts, scaler, kmeans):
    data_news = np.array([[amounts]])

    data_scaled = scaler.transform(data_news)

    cluster = kmeans.predict(data_scaled)

    if cluster[0] == 0:
        return "Comportement normal."
    else:
        return "Comportement anormal."
