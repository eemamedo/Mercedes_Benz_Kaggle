import xgboost as xgb
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score


def model (algorithm, X_train, X_test, y_train):
    
    if algorithm == 'XGBoost':
        kf = KFold (n_splits = 3, random_state = 24, shuffle = True)

        r2_scores_test = []
        r2_scores_train = []
        for train_index, test_index in kf.split(X_train):
            X_train_folded, X_test_folded = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_folded, y_test_folded = y_train.iloc[train_index], y_train.iloc[test_index]
            y_test_folded=y_test_folded.values
            y_train_folded = y_train_folded.values
          
            xgb_params = {'n_trees': 600,
                           'eta': 0.0025,
                           'max_depth': 5, #percentage of samples used per tree. low value can lead to underfitting
                           'subsample': 0.85,
                           'objective': 'reg:linear', #determines the loss function to be used:  reg:linear for regression problems,
                                                    # reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.
                           'eval_metric': 'rmse',
                           'base_score': np.mean(y_train.values)}                           
            
            dtrain = xgb.DMatrix (X_train_folded, y_train_folded)
            dtest = xgb.DMatrix(X_test_folded)
            xgb_model = xgb.train (xgb_params, dtrain, num_boost_round = 1500) 
            
            xgb_pred_train = xgb_model.predict(dtrain)
            r2_score_train = r2_score(y_train_folded, xgb_pred_train)
            r2_scores_train.append(r2_score_train)
            
            xgb_pred_test = xgb_model.predict(dtest)
            r2_score_test = r2_score(y_test_folded, xgb_pred_test)
            r2_scores_test.append(r2_score_test)
            
             #checking feature importance
#            print (xgb_model.feature_importances_)
#            plt.figure(figsize=(12,25))
                
#            imp = pd.DataFrame({'Feature': X_train_folded, 'Importance':xgb.feature_importances_}).sort_values('Importance')
#            plt.barh(range(len(X_train_folded)), imp.Importance)
#            plt.yticks(range(len(X_train_folded)), imp.Feature)
#            plt.show()  
            
        print ('\n R2 score for training: %.3f' % np.mean(r2_scores_train), 
               '\n R2 score for testing: %.3f' % np.mean(r2_scores_test))
        
    if algorithm == 'ElasticNetCV':
        # l1_ration, if 0 -> Ridge, if 1-> Lasso, if between 0 and 1, mix of two
        # alpha, regulation of amount of penalty applied
       
        kf = KFold (n_splits = 3, random_state = 24, shuffle = True)
        r2_scores_test = []
        r2_scores_train = []
        for train_index, test_index in kf.split(X_train):
            X_train_folded, X_test_folded = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_folded, y_test_folded = y_train.iloc[train_index], y_train.iloc[test_index]
    
            encv_model = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1], tol = 0.0001)
            encv_model.fit(X_train_folded,y_train_folded) 
            r2_score_train = r2_score (y_train_folded, encv_model.predict(X_train_folded))
            r2_scores_train.append(r2_score_train)
            r2_score_test = r2_score(y_test_folded, encv_model.predict(X_test_folded))
            r2_scores_test.append(r2_score_test)

        print ('\n R2 score for training: %.3f' % np.mean(r2_scores_train), 
               '\n R2 score for testing: %.3f' % np.mean(r2_scores_test))
    
    
    if algorithm == 'RandomForestRegressor':
        kf = KFold (n_splits = 3, random_state = 24, shuffle = True)
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
        
    if algorithm == 'DecisionTreeRegressor':
        kf = KFold (n_splits = 3, random_state = 24, shuffle = True)
        r2_scores_test = []
        r2_scores_train = []
        for train_index, test_index in kf.split(X_train):
            X_train_folded, X_test_folded = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_folded, y_test_folded = y_train.iloc[train_index], y_train.iloc[test_index]
    
            dt_model = BaggingRegressor (base_estimator=DecisionTreeRegressor(max_depth=4),
                           n_estimators=20,
                           bootstrap=True,
                           oob_score=True)
            dt_model.fit(X_train,y_train)
            r2_score_train = r2_score (y_train_folded, dt_model.predict(X_train_folded))
            r2_scores_train.append(r2_score_train)
            r2_score_test = r2_score(y_test_folded, dt_model.predict(X_test_folded))
            r2_scores_test.append(r2_score_test)
        
        print ('\n R2 score for training: %.3f' % np.mean(r2_scores_train), 
               '\n R2 score for testing: %.3f' % np.mean(r2_scores_test))

  

