import openml
import pickle
import logging
import pandas as pd
import numpy as np

from dataprep import features
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

def read_open_ml_data(id=24):
   dataset = openml.datasets.get_dataset(id)
   return dataset.get_data()[0]


def prepare_dataset():
   df = read_open_ml_data()
   columns = [x.replace("-", "_") for x in df.columns]
   columns = [x.replace("%3F", "") for x in columns]
   df.columns = columns
   # Map feature names
   for key, value in features.items():
    df[key] = df[key].map(value)
   # If mushroom is poisonous, then set it to 1
   df['class'] = np.where(df['class']=="p", 1, 0)
   # Remove column with missing values
   cols_to_keep = [x for x in df.columns if x not in ["stalk_root"]]
   df = df[cols_to_keep].copy()
   return df

def split_dataset(df):
   df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
   return df_full_train, df_test

def get_feats_and_target(df):
    y_train = df["class"].values
    del df["class"]
    return y_train, df

def transform_data(df):
    dicts = df.to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)
    return X, dv

def write_artifacts(dv, model):
    with open('./artifacts/model.bin', 'wb') as f_out:
       pickle.dump((dv, model), f_out)

def train_model():
   logger.info("Preparing raw data")
   df = prepare_dataset()
   logger.info("Splitting train vs. test")
   df_full_train, df_test = split_dataset(df)
   y_train, X_train = get_feats_and_target(df_full_train)
   logger.info("Model training")
   X_train, dv = transform_data(X_train)
   model = LogisticRegression(solver='liblinear', C=1, max_iter=1000, random_state=42)
   model.fit(X_train, y_train)
   logger.info("Writting Model")
   write_artifacts(dv, model)
   logger.info("Writing test data")
   df_test.to_parquet("./artifacts/df_test.parquet")

if __name__ == '__main__':
   train_model()