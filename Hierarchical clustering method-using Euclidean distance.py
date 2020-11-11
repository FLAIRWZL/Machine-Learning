#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn import cluster

data = pd.read_csv('qdata.csv')
X = data.drop('label',axis=1)
Y = data.label

from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca = pca.fit(X)
pca.explained_variance_ratio_

ht = pca.transform(X)
ht = pd.DataFrame(ht,columns=['x','y'])


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

ax = sns.scatterplot(x="x", y="y", data=ht)

clst = cluster.AgglomerativeClustering(n_clusters=4,linkage='average',affinity='euclidean')
predicted = clst.fit_predict(ht)

pre = pd.Series(predicted)

g = pd.concat([ht,pre],axis=1)

g1 = g.sort_values(by=0)

g1.columns = ['x','y','type']

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,8),dpi=500)
ax = sns.scatterplot(x="x", y="y",hue="type",data=g1)
plt.savefig('散点')
plt.show()

data.loc[g1[g1['type']==0].index].to_csv('qdata1.csv')


data.loc[g1[g1['type']==1].index].to_csv('qdata2.csv')


data.loc[g1[g1['type']==2].index].to_csv('qdata3.csv')


data.loc[g1[g1['type']==3].index].to_csv('qdata4.csv')





