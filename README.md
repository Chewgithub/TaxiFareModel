# Introduction
This repository attempts to predict the taxi fare in new york city based on the information of past fare provided by ([Kaggle New York City Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction))
<td align="center"><img src="https://jovian.ai/api/gist/d6abb03b10c94e8ab4901cd9e0a47a1e/preview/25bc8c7d660c4baea36cbfc071730f68" width="400px" height="250px"></td>

# Dataset and Features
A total of 1,000,000 taxi fare example were provided by the Kaggle dataset. Features provided includes pickup and dropoff coordinates, date, time, and number of passengers. (Note: However, due to the constraint of memory on personal computer, only 100,000 data point was used)

# Method
1. Feature Engineering:
- Converting datetime to date of week, hour, month and year.
- distance between pickup and dropoff point.
- distance between pickup point and center of new york city (40.7141667, -74.0063889).

2. Modelling
Tried out different model(sklearn version 0.22):
- Linear Model
- Ridge Model
- Random Forest Regressor
- Gradient Boosting (Best Model)

3. Deployment
- Isolate environment through Docker container, deploy container through Google CloudRun. API is accessible through this ([link](https://taxifaremodelapi-gf34ldcmyq-de.a.run.app/docs))

4. Frontend
- Frontend of this prediction can be access through ([here](https://chewtaxifarepredict.herokuapp.com/))
- The ([link](https://github.com/Chewgithub/TaxiFareWebsite)) repository of frontend application
<td align="center"><img src="https://github.com/Chewgithub/TaxiFareModel/blob/master/taxifarefrontend.JPG" width="700px" height="800px"></td>
