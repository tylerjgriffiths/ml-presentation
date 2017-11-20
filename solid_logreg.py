#!/usr/bin/env python

## CONFIGURATION
bradley_file = "/home/tjg/Development/Data/Bradley/bradleymp.csv"

## Import Data() class.
from data import Data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

## Convert float MP to simple boolean: 1 if the compound
## melts above 20 C, 0 if it melts below 20 C.
for row in fps:
    row[1] = int(row[1]>20)
    
X = np.array([row[0] for row in fps])
y = np.array([row[1] for row in fps])

## Split into 70:10:20 train:dev:test.
X_train,X_dev,X_test = np.split(X,[int(0.7*len(X)),int(0.8*len(X))])
y_train,y_dev,y_test = np.split(y,[int(0.7*len(y)),int(0.8*len(y))])

print("X trn:dev:tst={}:{}:{}".format(len(X_train),len(X_dev),len(X_test)))
print("y trn:dev:tst={}:{}:{}".format(len(y_train),len(y_dev),len(y_test)))

## Create a new SVC classifier and fit the data.

print("Fitting data...")

logreg_cls = LogisticRegression()
training_start_time = datetime.datetime.now().replace(microsecond=0)
logreg_cls.fit(X_train, y_train)
training_end_time = datetime.datetime.now().replace(microsecond=0)
print("Model trained in {}".format(training_end_time-training_start_time))

## Get the predictions.
print("Predicting...")
logreg_y_pred = logreg_cls.predict(X_dev)

## Calculate and print out some metrics.
acc = accuracy_score(y_dev, logreg_y_pred)
prec = precision_score(y_dev, logreg_y_pred)
rec = recall_score(y_dev, logreg_y_pred)
f1 = f1_score(y_dev, logreg_y_pred)
print("Accuracy: {}".format(acc))
print("Precision: {}".format(prec))
print("Recall: {}".format(rec))
print("F1 score: {}".format(f1))


