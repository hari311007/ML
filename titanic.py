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
knn=KNeighborsClassifier(n_neighbors=5)#when n=1 then it would be 0.962(not 1)(hint:the closest member to any point is the point itself:distance=0) symbolizes that there is duplicate data for all 4 columns

#scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()#here scaler is a object it contains entities of mean and standard deviation
X_scaler_train=scaler.fit_transform(X_train)
X_scaler_test=scaler.transform(X_test)
knn.fit(X_scaler_train,y_train)
print("knn classifer account:",round(knn.score(X_scaler_test,y_test),3))#score gets the ouput for its predicted y compares it with true y values and gives accuracy %
#alternate approach to knn.score(X_scaler_test,y_test)
y_pred=knn.predict(X_scaler_test)

from sklearn.metrics import accuracy_score
print("alternative approach of knn output: ",accuracy_score(y_test,y_pred))

#logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_scaler_train,y_train)
y_pred2=model.predict(X_scaler_test)
print("logistic regression: ",accuracy_score(y_test,y_pred2)) # comparison of logistic regression answer with real test output

#decision tree classifier:
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_scaler_train,y_train)
print("decision tree classifier output: ",round(dt.score(X_scaler_test,y_test),3))
#random forest classifier:
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=42)
rf.fit(X_scaler_train,y_train)
print("random forest classifier: ",round(rf.score(X_scaler_test,y_test),3))

#confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
y_pred1=rf.predict(X_scaler_test)
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))


