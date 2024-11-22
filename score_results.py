import logging
import pandas as pd
import requests
from train import get_feats_and_target

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()


def sample_test_data(n):
    df_test = pd.read_csv("./artifacts/df_test.csv").sample(n, random_state=42)
    y_test, X_test = get_feats_and_target(df_test)
    return y_test, X_test


def score_test_data(client):
    url = 'http://localhost:8787/predict' ## this is the route we made for prediction
    response = requests.post(url, json=client) ## post the customer information in json format
    result = response.json() ## get the server response
    print(result)

y_test, X_test = sample_test_data(n=5)
records = X_test.to_dict(orient='records')
for i, row in enumerate(records):
    logger.info(f"Scoring observation {i} with label {y_test[i]}")
    score_test_data(row)