#import data

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#gather data
df=pd.read_csv('data/concrete_data.csv')
features = df.drop(['strength'], axis=1)
target = df.strength

# make a case for storing predicted data
prediction_stat = np.zeros_like(features.columns)
prediction_stat = features.mean().values.reshape(1,8)
print(prediction_stat)
CEMENT_IDX = 0
AGE_IDX = 7

# model building
regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)
rsquared =regr.score(features, target)
mse = mean_squared_error(y_true=target, y_pred=fitted_vals)
rmse = np.sqrt(mse)

#prediction function
def get_strength_estimate(age):
    prediction_stat[0][AGE_IDX] = age
    
    strength = regr.predict(prediction_stat)[0]

    return strength