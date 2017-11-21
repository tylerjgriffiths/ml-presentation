#!/usr/bin/env python

## CONFIGURATION
bradley_file = "/home/tjg/Development/Data/Bradley/bradleymp.csv"
shape = (2048)

## Import Data() class.
from data import Data

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import numpy as np

import datetime

print("Loading data...")

## Create a new Data() object and read in the Bradley CSV.
d=Data()
d.read_bradley_from_csv(bradley_file)

## Get the data with molecules as fingerprints, using the
## default (Morgan) algorithm. Separate into separate
## X and y numpy arrays.
print("Creating fingerprints...")
fps = d.get_bradley_fps()
X = np.array([row[0] for row in fps])
y = np.array([row[1] for row in fps])

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

## Split into 70:10:20 train:dev:test.
X_train,X_dev,X_test = np.split(X,[int(0.7*len(X)),int(0.8*len(X))])
y_train,y_dev,y_test = np.split(y,[int(0.7*len(y)),int(0.8*len(y))])

print("X trn:dev:tst={}:{}:{}".format(len(X_train),len(X_dev),len(X_test)))
print("y trn:dev:tst={}:{}:{}".format(len(y_train),len(y_dev),len(y_test)))

## Create a new MLPRegressor and fit the data.

print("Fitting data...")
print("Shape: {}".format(shape))

mlpr_cls = MLPRegressor(hidden_layer_sizes=shape)
training_start_time = datetime.datetime.now().replace(microsecond=0)
mlpr_cls.fit(X_train, y_train)
training_end_time = datetime.datetime.now().replace(microsecond=0)
print("Model trained in {}".format(training_end_time-training_start_time))

## Get the predictions.
print("Predicting...")
mlpr_y_pred = mlpr_cls.predict(X_dev)

## Calculate and print out some metrics.
mae = mean_absolute_error(y_dev, mlpr_y_pred)
print("Mean absolute error: {}".format(mae))

rms = mean_squared_error(y_dev, mlpr_y_pred)
print("Mean squared error: {}".format(rms))

rsq = r2_score(y_dev, mlpr_y_pred)
print("R2 score: {}".format(rsq))
