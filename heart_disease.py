import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv("heart.csv")

print(df.head())

X=df.iloc[:, :-1].values
y=df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#dtc

dtc=DecisionTreeClassifier(random_state = 0)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print(y_pred)

#naive bayes

gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)

print(pred)
