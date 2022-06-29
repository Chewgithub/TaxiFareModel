import os
from google.cloud import storage
from termcolor import colored
from TaxiFareModel.params import BUCKET_NAME, MODEL_NAME





def storage_upload(model, rm=False):
    '''
    Upload model to google cloud bucket
    '''

    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name =  f'{model}model.joblib'
    storage_location = f"models/{MODEL_NAME}/{local_model_name}"

    blob = client.blob(storage_location)
    blob.upload_from_filename(f'{model}model.joblib')
    print(colored(f"=> {model}model.joblib uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    if rm:
        os.remove(f'{model}model.joblib')
