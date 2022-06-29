import pandas as pd
import os

'''
path for model training, download the dataset and replace the path if you
wish to rerun the script
'''
current = os.path.abspath(__file__)
path=os.path.abspath(os.path.join(current,'../../raw_data/train_1k.csv'))


def get_data():
    '''pandas read_csv file'''
    df = pd.read_csv(path)
    return df


def clean_data(df, test=False):
    '''
    Data cleaning
    '''
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df
