# behavior_tansaction

This project is a FastAPI application that uses a machine learning model to predict whether a transaction behavior is **normal** or **abnormal** based on transaction amount. It separates the data loading, model training, and FastAPI application for modularity.

## Architecture project

- `data_loader.py`: This file contains the logic for loading the data, preprocessing, and training the machine learning model.
- `model.py`: The model prediction logic is placed here, which is used to predict whether a transaction is normal or abnormal.
- `main.py`: The FastAPI app runs from this file, exposing an endpoint for prediction.
- `data.csv`: The file that contains data.

## Prerequisites

Make sure you have the following installed:

- Python 3.8+
- `pip` for installing Python packages

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repository/transaction-prediction-api.git
   cd transaction-prediction-api

2. **Create and activate a virtual environment**:

   ```bash
    python -m venv env
    source env/bin/activate   # For macOS/Linux
    .\env\Scripts\activate    # For Windows


3. **Install the required dependencies**:
    ```bash
   pip install -r requirements.txt

5. **Run the FastAPI application**:
    ```bash
   uvicorn app:app --reload

    url : http://127.0.0.1:8000/docs


  

