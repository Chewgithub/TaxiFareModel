import os
from math import sqrt

import joblib
import pandas as pd
from TaxiFareModel.params import MODEL_NAME,BUCKET_NAME,BUCKET_TEST_DATA_PATH
from google.cloud import storage


PATH_TO_LOCAL_MODEL = 'RandomForestRegressor()model.joblib'


def get_test_data(nrows, data="local"):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"  # ⚠️ to test from actual KAGGLE test set for submission

    if data == "local":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TEST_DATA_PATH}")

    return df


def download_model( bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/{}/{}'.format(
        MODEL_NAME,
        'RandomForestRegressor()model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('RandomForestRegressor()model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('RandomForestRegressor()model.joblib')
    if rm:
        os.remove('RandomForestRegressor()model.joblib')
    return model


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


# This is used in result submission for kaggle website
def generate_submission_csv(nrows, kaggle_upload=False):
    df_test = get_test_data(nrows)
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == '__main__':

    nrows = 100
    generate_submission_csv(nrows, kaggle_upload=False)
