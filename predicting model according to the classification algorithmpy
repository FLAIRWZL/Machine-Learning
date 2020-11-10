# -*- coding: utf-8 -*-
import  pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
warnings.filterwarnings('ignore')

np.set_printoptions(suppress=True)


df=pd.read_csv("D:/polyu/code/qdata1.csv",sep=',')
df=df.dropna()
df=df[[  'N',  'NI', 'NB', 'AL', 'FE','MO',  'CO', 'SI', 'CR', 'CU', 'property:Stacking fault energy (mJ/m^2)']]
df=df.rename(columns={'property:Stacking fault energy (mJ/m^2)':'target'})

df=pd.concat([df,df],axis=0)
df=pd.concat([df,df],axis=0)
df=pd.concat([df,df],axis=0)
df=shuffle(df)

print(df.shape)
x_target=df['target'].values
print(df.columns)
X=df[[ 'N', 'NI', 'NB', 'AL', 'FE', 'MO',  'CO', 'SI', 'CR', 'CU']].values
y_1=df['target'].values
label_encoder =LabelEncoder()
y = label_encoder.fit_transform(y_1)
lable_dic={}
for key ,value in zip(x_target,y):
    lable_dic[str(value)]=str(key)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
#
Classifiers = [["Random Forest", RandomForestClassifier()],
               ["Naive Bayes", GaussianNB()],
               ["Decision Tree", DecisionTreeClassifier()]
               ]
#
Classify_result = []
names = []
prediction = []

def change(x):
    try:
        return lable_dic[x]
    except Exception as e:
        pass
for name, classifier in Classifiers:
    print("name=",name)
    classifier = classifier
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy= accuracy_score(y_test, y_pred,)

    recall = recall_score(y_test, y_pred,average='weighted')

    f1 = f1_score(y_test, y_pred,average='weighted')
    print('accuracy: ', accuracy,'recall:',recall,'f1:',f1)

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def Plot_func(pred_y, test_y, R,rmse, model_name,w_plot = 'False'):
    print(R,rmse)
    model = LinearRegression()
    model.fit(test_y.reshape(test_y.shape[0],1), pred_y.reshape(pred_y.shape[0],1))
    y = model.predict(test_y.reshape(test_y.shape[0],1))
    if w_plot=='True':
        plt.figure()
        plt.scatter(test_y, pred_y, c='r')
        plt.plot(test_y, y)
        plt.grid(True)
        plt.xlabel('Measured SFE (mJ/m²)')
        plt.ylabel(f'{model_name}@ All Predicted Value')
        plt.text(10, 220, s=f'R={np.round(R,4)}', fontsize='x-large')
        plt.text(10, 200, s=f'RMSE={np.round(rmse, 4)}', fontsize='x-large')
        plt.show()

kf = KFold(n_splits=10,shuffle=False)
Train_index = []
for i,(train_index,test_index) in enumerate(kf.split(X)):
    Train_index.append(train_index)
M1e1, M1e2 = [],[]
M2e1, M2e2 = [],[]
M3e1, M3e2 = [],[]
for i in range(10):
    index = [0,1,2,3,4,5,6,7,8,9]
    index.remove(i)
    train_index = np.array([]).astype('int32')
    for j in index:
        train_index = np.hstack((train_index, Train_index[j]))
    X_train = X[train_index]
    y_train = y[train_index]

    est1 = GaussianNB()
    est1.fit(X_train,y_train)
    y_pred=est1.predict(X_test)
    M1e1.append(pearsonr(y_pred,y_test)[0])
    M1e2.append(RMSE(y_pred,y_test))

    est2 = RandomForestClassifier()
    est2.fit(X_train,y_train)
    y_pred=est2.predict(X_test)
    M2e1.append(pearsonr(y_pred,y_test)[0])
    M2e2.append(RMSE(y_pred,y_test))

    est3 = DecisionTreeClassifier()
    est3.fit(X_train,y_train)
    y_pred=est3.predict(X_test)
    M3e1.append(pearsonr(y_pred, y_test)[0])
    M3e2.append(RMSE(y_pred, y_test))
y_pred=est1.predict(X_test)
Plot_func(y_pred, y_test, np.mean(M1e1), np.mean(M1e2), 'GaussianNB', 'True')
y_pred=est2.predict(X_test)
Plot_func(y_pred, y_test, np.mean(M2e1), np.mean(M2e2), 'RandomForest', 'True')
y_pred=est3.predict(X_test)
Plot_func(y_pred, y_test, np.mean(M3e1), np.mean(M3e2), 'DecisionTree', 'True')
# Plot_func(y_pred, y_test, 'RandomForest')
# Plot_func(y_pred, y_test, 'DecisionTree')
