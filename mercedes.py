# url: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing#description

##### DESCRIPTION OF THE COMPETION #####
# Daimler asks to tackle the curse of dimensionality and reduce the time that cars spend on the test bench.
# Competitors will work with a dataset representing different permutations of Mercedes-Benz car features to
# predict the time it takes to pass testing. Winning algorithms will contribute to speedier testing, resulting
# in lower carbon dioxide emissions without reducing Daimler’s standards.

# Submissions are evaluated on the R^2 values, also known as coefficient of determination.
# For each ID, you must predict the 'y' variable; the time that it takes for a new car to pass testing



##### DESCRIPTION OF THE DATA #####
# This dataset contains an anonymized set of variables, each representing a custom feature in a Mercedes Car
# For example, a variable could be 4WD, added air suspension, or a head-up display.
# The ground truth is labeled ‘y’ and represents the time (in seconds) that the car took to pass testing
# for each variable.
# train.csv - the training set
# test. csv - the test set, you must predict the 'y' variable for the 'ID's in this file

# Loading libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import xgboost as xgb

from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from model import model

# Load Train and Test Data
train_data = pd.read_csv('train.csv', sep = ',')
X_test = pd.read_csv('test.csv', sep = ',')
X_train = train_data.drop('y', axis = 1) # drop y from the train and assign that y (target) to y_train
y_train = pd.DataFrame(train_data ['y'])

##### Data Exploration Stage #####
# Plot y variable
plt.figure(figsize = (8,8))
plt.scatter(range(len(train_data)), np.sort(y_train))
plt.xlabel ('index')
plt.ylabel ('y (target) values')

# It is easy to see that there is a single outlier. It is possible to visualize the same plot for the target variable
# using a different plot
plt.figure(figsize = (8,8))
plt.scatter(range(train_data.shape[0]), np.sort(train_data['y']))
plt.xlabel ('index')
plt.ylabel ('Sorted y (target) values')

# An outlier might affect the performance of an algorithm: It is the best to either remove it, or impute it.
# From the graphs, it is obvious that the y value for that outlier is approximately 270. To be on the safe side, let's
# enforce the condition for the outlier to be >190; the value it will definitely satisfy. Instead of removing it,
# the outlier will be imputed using the mean-value approach.
y_train.loc[y_train['y']>190] = y_train ['y'].mean ()

# Plot again
sorted_vals = sorted (y_train.values)
plt.figure(figsize = (8,8))
plt.scatter(range(len(train_data)), sorted_vals)
plt.xlabel ('index')
plt.ylabel ('Sorted y (target) values')

# As it can be seen, no outliers are present


# Let's explore the training data: X_train (X_test gives the same results)
dict (Counter(X_train.dtypes))


# We can see that 369 features are int64.
# There are 8 categorical features, namely; ID, X0, X1, X2, X3, X4, X5, X6, X7, X8.
# Of course, I can do feature reduction at this point, but I want to see if I can manually reduce features.
# A good approach is to get rid of the features that have no variations
unique_values_dict = {}
for col in X_train.columns:
    if col not in ['ID', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']:
        unique_value = str(np.sort(X_train[col].unique()).tolist())
        tlist = unique_values_dict.get(unique_value, [])
        tlist.append(col)
        unique_values_dict[unique_value] = tlist [:]


# I want to print them to see if I can drop any
for unique_value, columns in unique_values_dict.items():
    print ('\nColumns that have unique values are ', unique_value )
    print (columns)


# There are 12 features that have no variance at all. Drop them
X_train = X_train.drop(columns=unique_values_dict['[0]'], axis=1)
X_train = X_train.drop(['ID'], axis = 1)
X_test = X_test.drop(columns=unique_values_dict['[0]'], axis=1)
X_test = X_test.drop(['ID'], axis = 1)

# There are 8 categorical features, and they need to be converted to numerical representation. The two common methods
# are LabelEncoder or OneHotEncoder. The problem with OneHotEncoder is that it ttakes a column which has categorical
# data, and then splits that column into multiple columns. The numbers are replaced by 1 and 0, depending on which
# column has what value. So, if we have France, Germany and Spain, we will get 3 new columns.
# LabelEncoder though, replaces categorical data with values from 0 to n_classes - 1

from sklearn.preprocessing import LabelEncoder
categorical_features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
for cf in list(X_train.columns):
    if cf in categorical_features:
        lbl = LabelEncoder()
        X_train[cf] = lbl.fit_transform(X_train[cf].values)
        X_test[cf] = lbl.fit_transform(X_test[cf].values)

# We can also take a look at what features are important for target prediction.
# For that we use embedded method feature selection called Random Forest feature selection

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

# We can drop the features that are not as important as the others in predicting the target function
X_train = X_train.drop(order_features[250:],axis = 1)
X_test = X_test.drop(order_features[250:],axis = 1)

print (model('RandomForestRegressor', X_train,X_test,y_train))
