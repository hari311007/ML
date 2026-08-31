import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('train.csv')
df.head()
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin',axis=1,inplace=True)
df['Sex']=(df['Sex']=="male").astype(int)
df['Embarked']=df['Embarked'].map({'S':0,'C':1,'Q':2})
from sklearn.model_selection import train_test_split
X=df[['Pclass','Embarked','Sex','Age','Fare']]
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)#when n=1 then it would be 0.962(not 1)(hint:the closest member to any point is the point itself:distance=0) symbolizes that there is duplicate data for all 4 columns
knn.fit(X_train,y_train)
print(round(knn.score(X_train, y_train), 3))
