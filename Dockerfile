FROM python:3.8.12-buster

COPY GradientBoostmodel.joblib /GradientBoostmodel.joblib
COPY TaxiFareModel /TaxiFareModel
COPY api /api
COPY predict.py /predict.py
COPY requirements.txt /requirements.txt
COPY wagon-bootcamp-347403-d06e1020e122.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
