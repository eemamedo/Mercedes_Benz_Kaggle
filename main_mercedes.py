import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter

from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)
import random as rn
rn.seed(1)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
import keras as k
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold

#Model evaluation
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

# Load Train and Test Data
train_data = pd.read_csv('train.csv', sep = ',')
X_test = pd.read_csv('test.csv', sep = ',')
X_train = train_data.drop('y', axis = 1) 
y_train = pd.DataFrame(train_data['y'])

y_train.loc[y_train['y']>190] = y_train ['y'].mean ()
unique_values_dict = {}
for col in X_train.columns:
    if col not in ['ID', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']:
        unique_value = str(np.sort(X_train[col].unique()).tolist())
        tlist = unique_values_dict.get(unique_value, [])
        tlist.append(col)
        unique_values_dict[unique_value] = tlist [:]

X_train = X_train.drop(columns=unique_values_dict['[0]'], axis=1)
X_train = X_train.drop(['ID'], axis = 1)
X_test = X_test.drop(columns=unique_values_dict['[0]'], axis=1)  
X_test = X_test.drop(['ID'], axis = 1)  

from sklearn.preprocessing import LabelEncoder
categorical_features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

for cf in list(X_train.columns):
    if cf in categorical_features:
        lbl = LabelEncoder()
        X_train[cf] = lbl.fit_transform(X_train[cf].values)
        X_test[cf] = lbl.fit_transform(X_test[cf].values)
        
feature_labels = X_train.columns
forest_model = RandomForestRegressor(n_estimators = 1000, random_state = 0, n_jobs = -1) # -1 mean that trees are constructed in parallel
forest_model.fit (X_train, y_train)
importance = forest_model.feature_importances_  

indices = np.argsort(importance)[::-1]
order_features = []
order_importances = []
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feature_labels[f], importance[indices[f]]))
    order_features.append(feature_labels[f])
    order_importances.append(importance[indices[f]])

X_train = X_train.drop(order_features[243:],axis = 1)
X_test = X_test.drop(order_features[243:],axis = 1)

# Run with RFRegressor
kf = KFold (n_splits = 5, random_state = 42, shuffle = True)
r2_scores_test = []
r2_scores_train = []
for train_index, test_index in kf.split(X_train):
    X_train_folded, X_test_folded = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_folded, y_test_folded = y_train.iloc[train_index], y_train.iloc[test_index]

    rf_model = RandomForestRegressor (n_estimators=20, oob_score=True, bootstrap=True, max_depth = 4)
    rf_model.fit(X_train_folded,y_train_folded)
    r2_score_train = r2_score (y_train_folded, rf_model.predict(X_train_folded))
    r2_scores_train.append(r2_score_train)
    r2_score_test = r2_score(y_test_folded, rf_model.predict(X_test_folded))
    r2_scores_test.append(r2_score_test)

print ('\n R2 score for training: %.3f' % np.mean(r2_scores_train), 
       '\n R2 score for testing: %.3f' % np.mean(r2_scores_test))

print (r2_scores_test)

