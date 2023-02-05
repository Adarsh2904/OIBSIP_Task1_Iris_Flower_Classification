#total imported Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("Iris.csv")

print(df)

print(df.head(10))
print(df.tail(10))
df.info()
df.describe()
print(df.shape)
print(df.size)
print(df.columns)
print(df.isnull().sum())
dfinal =df.drop('Id', axis = 1)
print(dfinal)

dv=sb.pairplot(dfinal, hue = 'Species', palette = 'hls')
print(dv)

sb.lmplot( x="SepalLengthCm", y="SepalWidthCm", data=dfinal, fit_reg=False, hue='Species', legend=False)
pt.legend(loc='lower right')
rmv=pt.show()
print(rmv)

sb.lmplot( x="PetalLengthCm", y="PetalWidthCm", data=dfinal, fit_reg=False, hue='Species', legend=False)
pt.legend(loc='lower right')
rmv1=pt.show()
print(rmv1)

dfinal.hist()
histogram=pt.show()
print(histogram)


df.corr()
pt.figure(figsize=(8 ,4))
sb.heatmap(dfinal.corr(), annot=True, cmap='viridis')
heatmap=pt.show()
print(heatmap)

#model

xside=dfinal.drop(['Species'],1)
xside.head()


yside=dfinal['Species']
yside.head()

xtrain, xtest, ytrain, ytest = train_test_split(xside, yside, test_size= 0.4)

model=SVC()
model.fit(xtrain, ytrain)
predict = model.predict(xtest)

print(accuracy_score(ytest,predict)*100)
