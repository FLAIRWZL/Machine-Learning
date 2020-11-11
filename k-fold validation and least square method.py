#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_csv('qdata1.csv')


alphas1 = [0.000001,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,
     0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,
     0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
     0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
     0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
     1,2,3,4,5,6,7,8,9,10]


log_alphas = -np.log10(alphas1)

X = data.drop(['Unnamed: 0','label'],axis=1)
Y = data['label']


train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state=234)



def test_Lasso_alpha(train_x,train_y):
    alphas =alphas1
    coffs = []
    RMSE = []
    for i,alpha in enumerate(alphas):
        la=Lasso(alpha=alpha)
        la.fit(X,Y)
        pre=la.predict(X)
        q=np.sqrt(mean_squared_error(pre,Y))
        RMSE.append(q)
        
        coffs.append(list(la.coef_))
        
    return RMSE

RE = test_Lasso_alpha(X,Y)

RE=pd.DataFrame(RE)
log_alphas1 = pd.DataFrame(log_alphas)

he = pd.concat([RE,log_alphas1],axis=1)
he.columns=['x','y']

X=np.array(X)
Y=np.array(Y)

from sklearn.model_selection import KFold

n_splits=10
tr = []
te = []
foder = KFold(n_splits=n_splits,random_state=234,shuffle=True)
for train_index,test_index in foder.split(X,Y):
    tr.append(train_index)
    te.append(test_index)

ms=[]
for i in range(n_splits):
    train_x = X[tr[i]]
    train_y = Y[tr[i]]
    test_x=X[te[i]]
    test_y=Y[te[i]]
    alphas =  alphas1
    coffs = []
    RMSE = []
    scor =[]
    pr=[]
    for i,alpha in enumerate(alphas):
        la=Lasso(alpha=alpha)
        la.fit(train_x,train_y)
        scor.append(la.score(test_x,test_y))
        pre = la.predict(test_x)
        pr.append(pre)
        q=np.sqrt(mean_squared_error(pre,test_y))
        RMSE.append(q)
        coffs.append(list(la.coef_))
        

RM = pd.DataFrame(RMSE)
a1 = pd.DataFrame(log_alphas)

pp =pd.concat([RM,a1],axis=1)
pp.columns=['y','x']


import matplotlib.pyplot as plt

fig,ax= plt.subplots(figsize=(10,6),dpi=800)
ax.plot(pp['x'],pp['y'],'.-',label="10 fold cross validation",c="red")
ax.plot(he['y'],he['x'],'.-',label="All data",c="black")
plt.axvline(x=1.522879,ls="--",c="red")
plt.axvline(x=2.397940,ls="--",c="black")
plt.ylabel('RMSE')
plt.xlabel(r'$-\log_{10}({\alpha})$')
#plt.text(1.522879,12.478328,"The value of min",fontsize=15,verticalalignment="center")
#plt.text(2.397940,12.609438,"The value of min",fontsize=15)
plt.legend()
plt.savefig("All--fold")
plt.show()


pre_test = pd.concat([pd.DataFrame(pr[32]),pd.DataFrame(test_y)],axis=1)
pre_test.columns = ['pre','test_y']

x=np.arange(0,70,1)


y=x
y1=x + 10
y2=x - 10 

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5),dpi=200)
ax.scatter(pre_test['pre'],pre_test['test_y'],c='black',label=None)
ax.plot(x,y,'--',c="black")
ax.fill_betweenx(x,y1,y2,alpha=0.5,label="Error=\u00B110")
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.legend(title="RMSE=13.1")
plt.xlim(0,65)
plt.ylim(0,65)
plt.savefig("K-fold")
plt.show;

np.round(coffs[32],2)

X1 = data[['N','NI','NB','AL','FE','MO','CO','SI','CR','CU']]
Y1 = data.label


train_x1,test_x1,train_y1,test_y1 =train_test_split(X1,Y1,test_size=0.2,random_state=234) 


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR = LR.fit(train_x1,train_y1)
np.round(LR.coef_,2)

pred = LR.predict(test_x1)
ols_RMSE = np.sqrt(mean_squared_error(pred,test_y1))
ols_RMSE




