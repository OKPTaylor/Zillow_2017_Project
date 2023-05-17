import scipy.stats as stats
import pandas as pd
import os
import numpy as np

# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn stuff:
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import  mean_squared_error
from math import sqrt


import env

def regression_errors(y, yhat):
        '''
        Returns the following values:
        root mean squared error (RMSE) and r-squared (R2)
        '''
        #import
        
        #calculate r2
        r2 = r2_score(y, yhat)
        #calculate MSE
        MSE = mean_squared_error(y, yhat)
        #calculate RMSE
        RMSE = sqrt(MSE)
        
        return RMSE, r2

def test_poly_model(y_train, y_test, x_train_scaled,x_test_scaled):
    #polynomial regression
    #make polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=3)
    #fit and transform x_train_scaled
    x_train_scaled_pf = pf.fit_transform(x_train_scaled)
    #transform x_test_scaled
    x_test_scaled_pf = pf.transform(x_test_scaled)

    #fit to linear regression model
    #make the model
    pr = LinearRegression()
    #fit the model
    pr.fit(x_train_scaled_pf, y_train)
    #predict
    pred_pr = pr.predict(x_train_scaled_pf)
    #predict test
    pred_test_pr = pr.predict(x_test_scaled_pf)

    #evaluate pr
    regression_errors(y_train, pred_pr)
    rmse, r2 = regression_errors(y_test, pred_test_pr)

    metric_df = ['Polynomial Model Test', rmse, r2]
    print(metric_df)
   
   #plot actuals vs predicted
    baseline=299778.909266
    plt.figure(figsize=(16,8))
    plt.plot(y_test, y_test, color='green', label='Actual')
    plt.scatter(y_test, pred_test_pr, color='red', alpha=.5, label='Model 1: Polynomial')
    #plot basaeline
    plt.axhline(baseline, label='Baseline', color = 'blue')
    plt.legend()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

    #plot residuals
    plt.figure(figsize=(16,8))
    plt.scatter(y_test, pred_test_pr - y_test, color='green', label='Model 1: Polynomial', alpha=.5)
    plt.axhline(label="No Error")
    plt.legend()
    plt.xlabel('Actual')
    plt.ylabel('Residual/Error: Predicted - Actual')
    plt.title('Residual/Error: Predicted - Actual')
    plt.show()
    
    

          