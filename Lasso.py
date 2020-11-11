#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import numpy as np


a1 = pd.read_csv('qdata1.csv')

a = [0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,
     0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,
     0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
     0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
     0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
     1,2,3,4,5,6,7,8,9,10]



X = a1.drop(['Unnamed: 0','label'],axis=1)
Y = a1['label']



def test_Lasso_alpha(train_x,train_y):
    alphas =  a
    coffs = []
    for i,alpha in enumerate(alphas):
        la=Lasso(alpha=alpha)
        la.fit(train_x,train_y)
        coffs.append(list(la.coef_))
        
    return coffs

from sklearn.preprocessing import StandardScaler
sca = StandardScaler()
sca = sca.fit_transform(X)



XX = pd.DataFrame(sca)

g = test_Lasso_alpha(XX,Y)
qt = pd.DataFrame(g)

alphas = a
alps = -np.log10(alphas)

dt = pd.concat([qt,pd.Series(alps)],axis=1)

w = list(X.columns)
w.append('x')

dt.columns=w

import matplotlib.pyplot as plt

fig,ax= plt.subplots(figsize=(10,6),dpi=300)
ax.plot(dt['x'],dt['C'],'o-')

ax.plot(dt['x'],dt['N'],'v-')

ax.plot(dt['x'],dt['P'],'^-')
ax.plot(dt['x'],dt['S'],'.-')
ax.plot(dt['x'],dt['V'],'>-')
ax.plot(dt['x'],dt['NI'],'<-')
ax.plot(dt['x'],dt['NB'],'p-')
ax.plot(dt['x'],dt['AL'],'*-')
ax.plot(dt['x'],dt['TI'],'h-')
ax.plot(dt['x'],dt['FE'],'H-')
ax.plot(dt['x'],dt['HF'],'+-')
ax.plot(dt['x'],dt['MO'],'D-')
ax.plot(dt['x'],dt['MN'],'o-')
ax.plot(dt['x'],dt['CO'],'1-')
ax.plot(dt['x'],dt['SI'],'o-')
ax.plot(dt['x'],dt['CR'],'s-')
ax.plot(dt['x'],dt['CU'],'o-')


plt.axvline(x=0.39794,ls="--",c="black")
plt.axvline(x=0,ls="--",c="black")
plt.axvline(x=-0.602060,ls="--",c="black")
plt.axvline(x=0.221849,ls="--",c="black")
plt.axvline(x=-0.778151,ls="--",c="black")
plt.axvline(x=-0.301030,ls="--",c="black")
plt.axvline(x=0.045757,ls="--",c="black")
plt.axvline(x=-0.301030,ls="--",c="black")
plt.axvline(x=-0.698970,ls="--",c="black")
plt.axvline(x=2.045757,ls="--",c="black")
plt.axvline(x=-0.301030,ls="--",c="black")
plt.axvline(x=-0.602060,ls="--",c="black")
plt.axvline(x=0.397940,ls="--",c="black")

plt.ylabel('Coefficients')
plt.xlabel(r'$-\log_{10}({\alpha})$')
plt.ylim(-10,8)
plt.legend(ncol=3)
plt.savefig('第一簇')
plt.show();



