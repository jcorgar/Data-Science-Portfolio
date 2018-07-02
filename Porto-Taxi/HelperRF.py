# -*- coding: utf-8 -*-


import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def errores(model,start, end, df):
	s1=df.copy()
	scaler = StandardScaler()
	s1.columns = ["y"]
	for i in range(6, 25):
	    s1["lag_{}".format(i)] = s1.y.shift(i)
	s1["hour"] = s1.index.hour
        s1["weekday"] = s1.index.weekday
        s1['is_weekend'] = s1.weekday.isin([5,6])*1
	X=s1.dropna().drop(['y'], axis=1)
	y = s1.dropna().y
	X=scaler.fit_transform(X)
	forecast=model.predict(X[start:end])
	return mean_absolute_percentage_error(y[start:end],forecast)
