# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:47:10 2022

@author: 33o
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
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

# Upload appropriate table
#   get list of commodities 
def Table_Prep():
    df = pd.read_csv(r'C:\Users\33o\Documents\Python Scripts\data\master_oil_gas_table.csv')
    cols = list(df.columns.unique())
    df.drop([])
    df.drop(['n1000_1','n1000_2','n1000_3','n1000_4','year','FAF'],axis=1,inplace=True)

    return df,cols

def Model_Prep(df):
    
    #df = df[df['value in tons']!=0]
    df.drop(['gas','oil','ap','qp1','emp'],axis=1,inplace=True)
    x = df.drop('gas+oil',axis=1)
    y = df['gas+oil']
    
    #x = np.log(x)
    
    x_train, x_test, y_train, y_test = train_test_split\
        (x, y, test_size=0.25)
    
    le = LabelEncoder()
    #y_train = le.fit_transform(y_train)
    #y_test = le.fit_transform(y_test)
    

    scaler = preprocessing.StandardScaler().fit(x_train)
    #x_train = scaler.transform(x_train)
    #x_test = scaler.transform(x_test)
    
    return x_train,y_train,x_test,y_test
    
def Correlation(df):
    correlated = df.corr()
    return correlated

fuel_df, cols= Table_Prep()
correlated = Correlation(fuel_df)
x_train,y_train,x_test,y_test = Model_Prep(fuel_df)
    

#estimator = GradientBoostingRegressor(loss='squared_error')
#'learning_rate': [0.05,0.04,0.03,0.02],
#estimator = RandomForestRegressor()
#estimator = xgb.XGBRegressor()
estimator = KNeighborsRegressor()

parameters = {
#              'n_estimators' : [50,100,150],
#              'max_depth'    : [3,5],
#              'eta'          : [0.01,0.02],
              'n_neighbors'        : [3]
#              'colsample_bytree': [0.5],
#              'subsample'    : [0.5,1],
#              'reg_alpha'    : [1.1],
#              'reg_lambda'   : [1.1,1.3]
              }

fit_params={"early_stopping_rounds":70,
            'eval_metric': 'rmsle',
            "eval_set" : [[x_test, y_test]]}


clf = GridSearchCV(estimator,parameters,n_jobs=-1,cv=10)
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

#print('-'*30)
#print('Train log: '+str(mean_squared_log_error(y_train, y_p)))
 
print('-'*30)
print('Train RMSE: '+str(mean_squared_error(y_train, y_p,squared=(False))))

print('-'*30)
print('Train R^2: '+str(r2_score(y_train, y_p)))


print('-'*30)
print('Test MAE: '+str(mean_absolute_error(y_test, y_pred)))

#print('-'*30)
#print('Test log: '+str(mean_squared_log_error(y_test, y_pred)))

print('-'*30)
print('Test RMSE: '+str(mean_squared_error(y_test, y_pred,squared=(False))))

print('-'*30)
print('Test R^2: '+str(r2_score(y_test, y_pred)))


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(9,3.5))
fig.suptitle('Gas + Oil Analysis')

ax1.scatter(y_p,y_train)
ax1.set_ylabel('predicted gas+oil production (tons)')
ax1.set_xlabel('true gas+oil production (tons)')
ax1.set_xlim(-5000,125000)
ax1.set_ylim(-5000,125000)
ax1.title.set_text('Training Set')
b, a = np.polyfit([0,1], [0,1], deg=1)
xseq = np.linspace(0, 200000, num=100)
ax1.plot(xseq, a + b * xseq, color="k", lw=2.5,linestyle='--')
ax1.text(50000,15000,'R^2 = '+str(round(r2_score(y_train, y_p),4)),bbox=dict\
         (facecolor='c', alpha=0.5),size='large')
xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax1.get_xticks()/1000]
ax1.set_xticklabels(xlabels)    
ax1.set_yticklabels(xlabels)   

ax2.scatter(y_pred,y_test)
ax2.set_ylabel('predicted gas+oil production (tons)')
ax2.set_xlabel('true gas+oil production (tons)')
ax2.set_xlim(-5000,125000)
ax2.set_ylim(-5000,125000)
ax2.title.set_text('Test Set')
d, c = np.polyfit([0,1], [0,1], deg=1)
xseq = np.linspace(0, 200000, num=100)
ax2.plot(xseq, c + d * xseq, color="r", lw=2.5,linestyle='--')
ax2.text(50000,15000,'R^2 = '+str(round(r2_score(y_test, y_pred),4)),bbox=dict\
         (facecolor='r', alpha=0.5),size='large')
    
xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax2.get_xticks()/1000]
ax2.set_xticklabels(xlabels)
ax2.set_yticklabels(xlabels) 
    
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
#temp = pd.DataFrame({'true':y_test,'predicted':y_pred})
#temp.to_csv('predicted.csv')

    