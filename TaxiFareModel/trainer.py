from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS

from memoized_property import memoized_property
from  mlflow.tracking import MlflowClient
import mlflow

import joblib


class Trainer():
    def __init__(self, X, y, model):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.experiment_name="[SG] [Pasir Ris] [Chew Eng Yong] TaxiFarePrediction v1"
        self.model=model
        self.pipeline = self.set_pipeline()
        self.X = X
        self.y = y

        MLFLOW_URI = "https://mlflow.lewagon.ai/"
        self.mlflow_log_param('Model', self.model)
        self.mlflow_log_metric('RMSE', self.cross_v())
        self.save_model()

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])], remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                         ('Model', self.model)])
        return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return print(f'RMSE: {rmse}')

    def cross_v(self):
        score=cross_validate(self.pipeline, self.X,self.y,cv=5,
                             scoring=['neg_root_mean_squared_error'])
        return score["test_neg_root_mean_squared_error"].mean()


    def save_model(self):
        """ Save the trained model into a model.joblib file """
        return joblib.dump(self.run , f'{self.model}model')



    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == "__main__":
    df = clean_data(get_data()) # get data #clean_data
    y = df["fare_amount"]   # set X and y
    X = df.drop("fare_amount", axis=1)



    all_model=['linear','RandomForestRegressor','Ridge']
    for model in all_model:
        if model == 'linear':
            trainer = Trainer(X, y, LinearRegression()).cross_v()

        if model == 'RandomForestRegressor':
            trainer = Trainer(X, y, RandomForestRegressor()).cross_v()

        if model == 'Ridge':
            trainer = Trainer(X, y, Ridge()).cross_v()
