from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer, distance_to_center
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data,clean_data
from TaxiFareModel.gcp import storage_upload
from TaxiFareModel.params import MLFLOW_URI,experiment_name, BUCKET_NAME,BUCKET_TRAIN_DATA_PATH

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS

from memoized_property import memoized_property
from  mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
from google.cloud import storage

import joblib


class Trainer():
    def __init__(self, X, y, model, model_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.experiment_name=experiment_name
        self.model=model
        self.model_name=model_name
        self.pipeline = self.set_pipeline()
        self.X = X
        self.y = y

        #ML flow paramters and result logging
        self.MLFLOW_URI=MLFLOW_URI
        self.mlflow_log_param('Model', model_name)
        self.mlflow_log_param('Encoders', 'distance_to_center')
        self.mlflow_log_param('nrows', nrows)
        self.mlflow_log_metric('RMSE', self.cross_v())

        # gcloud model storage
        self.STORAGE_LOCATION = f'models/taxifare/{model_name}model.joblib'
        self.BUCKET_NAME = BUCKET_NAME


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        distc_pipe = Pipeline([
            ('dist_trans', distance_to_center()),
            ('stdscaler', StandardScaler())])

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance_center', distc_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
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
        '''
        Perform 5 fold cross validation
        '''
        score=cross_validate(self.pipeline, self.X,self.y,cv=5,
                             scoring=['neg_root_mean_squared_error'])
        score=abs(score["test_neg_root_mean_squared_error"].mean())

        return score

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        return joblib.dump(self.pipeline , f'{self.model_name}model.joblib')


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
    nrows=100000
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=nrows)

    df = clean_data(df) # get data #clean_data
    y = df["fare_amount"]   # set X and y
    X = df.drop("fare_amount", axis=1)


    small=['GradientBoost']
    all_model=['linear','RandomForestRegressor','Ridge','GradientBoost']
    for model in small:
        if model == 'linear':
            trainer=Trainer(X, y, LinearRegression(),model)
        if model == 'RandomForestRegressor':
            trainer = Trainer(X, y, RandomForestRegressor(),model)
        if model == 'Ridge':
            trainer = Trainer(X, y, Ridge(),model)
        if model == 'GradientBoost':
            trainer = Trainer(X, y, GradientBoostingRegressor(),model)
        trainer.run()
        trainer.save_model()
        trainer.cross_v()
        storage_upload(model,rm=False)
