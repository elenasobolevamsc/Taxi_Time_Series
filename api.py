import pandas as pd
import sklearn
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin

sklearn.set_config(transform_output='pandas')

app = FastAPI()


class MakeFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, max_lag):
        self.max_lag = max_lag

    def transform(self, X):
        X['dayofweek'] = X.index.dayofweek
        X['hour'] = X.index.hour

        for lag in range(1, self.max_lag + 1):
            X['lag_{}'.format(lag)] = X['num_orders'].shift(lag)

        X['rolling_mean'] = X['num_orders'].shift().rolling(self.max_lag).mean()

        X = X.drop('num_orders', axis=1)

        return X

    def fit(self, X, y=None):
        return self


class Dataframe(BaseModel):
    data: str


@app.post("/mod")
async def best_model(one_var: Dataframe):
    model_loaded = joblib.load('/best_model.pkl')
    df = pd.read_json(one_var.data, orient='split')
    pred = model_loaded.predict(df)
    return {"pred": pred.to_json(orient='split')}