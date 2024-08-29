from sklearn.base import BaseEstimator, TransformerMixin


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