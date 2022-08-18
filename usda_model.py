# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:35:00 2022

@author: Erick Orozco (33o)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import r2_score





# Upload appropriate table
#   get list of commodities 
def Table_Prep():
    df = pd.read_csv(r'C:\Users\33o\Documents\Python Scripts\data\master_USDA_table.csv')
    df['Pandemic'] = np.where(df['year']==2020, 1, 0)
    df.drop(df.index[(df['Program'] == 'SURVEY')&\
                               (df['year'] == 2017)], inplace=True)
    df.drop(['CV (%)'],axis=1,inplace=True)
    commodities = list(df[df['year']>2017]['Commodity'].unique())
    df = df[df.units == 'TONS']
    temp = []
    for com in commodities:
        if len(df[df['Commodity']==com])>200:
            temp.append(com)
    commodities = temp
    df.drop(['Program','year','winter tmin','winter tmax','spring tmin',\
                  'spring tmax','summer tmin','summer tmax','fall tmin',\
                      'fall tmax','State FIPS','County FIPS',\
                          'Ag District Code','Value','units',],axis=1,\
                 inplace=True)

    return df,commodities


def Model_Prep(df):
    
    #df = df[df['value in tons']!=0]

    x = df.drop('value in tons',axis=1)
    y = df['value in tons']
    
    #x = np.log(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    test_scaled = scaler.transform(x_test)

    return x_scaled,y_train,test_scaled,y_test

def Correlation(df,coms):
    target_corr = pd.DataFrame()
    correlated = {}
    for com in coms:
        temp_df = df[df['Commodity']==com].drop(['Commodity'],axis=1)   
        corr = temp_df.corr(method='pearson')
        target_corr[com] = corr['value in tons']
        correlated[com] = list(target_corr[com][target_corr[com].\
                                                abs()>.1].index)
    return correlated

usda_df, coms= Table_Prep()
correlated = Correlation(usda_df, coms)
x_train,y_train,x_test,y_test = Model_Prep(usda_df[usda_df['Commodity']==\
                coms[0]].drop(['Commodity'],axis=1)[correlated[coms[0]]])



#estimator = GradientBoostingRegressor(loss='squared_error')
#'learning_rate': [0.05,0.04,0.03,0.02],
#estimator = RandomForestRegressor()
estimator = xgb.XGBRegressor()


parameters = {
              'n_estimators' : [100],
              'max_depth'    : [3,5,7],
              'eta'          : [0.01],
#              'colsample_bytree': [0.5],
   #           'subsample'    : [0.5],
#              'reg_alpha'    : [1.1],
#              'reg_lambda'   : [1.1,1.3]
              }

fit_params={"early_stopping_rounds":70,
            'eval_metric': 'rmsle',
            "eval_set" : [[x_test, y_test]]}


clf = GridSearchCV(estimator,parameters,n_jobs=-1,cv=5)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
y_p = clf.predict(x_train)

for i in y_p:
    if i <0:
        print(i)
print('*'*30)
for i in y_pred:
    if i <0:
        print(i)

print('\n'+'-'*30)
print(clf.best_params_)

print('\n'+'-'*30)
print('Train MAE: '+str(mean_absolute_error(y_train, y_p)))

print('-'*30)
print('Train log: '+str(mean_squared_log_error(y_train, y_p)))
 
print('-'*30)
print('Train RMSE: '+str(mean_squared_error(y_train, y_p,squared=(False))))

print('-'*30)
print('Train R^2: '+str(r2_score(y_train, y_p)))


print('-'*30)
print('Test MAE: '+str(mean_absolute_error(y_test, y_pred)))

print('-'*30)
print('Test log: '+str(mean_squared_log_error(y_test, y_pred)))

print('-'*30)
print('Test RMSE: '+str(mean_squared_error(y_test, y_pred,squared=(False))))

print('-'*30)
print('Test R^2: '+str(r2_score(y_test, y_pred)))

plt.scatter(y_pred,y_test)
plt.title(str(clf.best_params_))
plt.ylabel('predicted')
plt.xlabel('true')
plt.xlim(0,700)
plt.ylim(0,700)
plt.text(600,800,'Test R^2 = '+str(r2_score(y_test, y_pred)),bbox=dict\
         (facecolor='r', alpha=0.3))
plt.text(600,870,'Train R^2 = '+str(r2_score(y_train, y_p)),bbox=dict\
         (facecolor='c', alpha=0.5))
#temp = pd.DataFrame({'true':y_test,'predicted':y_pred})
#temp.to_csv('predicted.csv')
