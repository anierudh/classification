from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing
import pandas as pd
import numpy as np
train=pd.read_csv("train.csv")
td=pd.DataFrame(train)
#print (td)

test=pd.read_csv("test.csv")
td1=pd.DataFrame(test)
#print (td1)
td.isnull().any()
td['Age']=td['Age'].fillna(0,inplace=False)

td['Cabin']=td['Cabin'].fillna(0,inplace=False)



col=['Name','Ticket','Cabin','PassengerId']
td=td.drop(col,axis=1)
#td=td.drop('Name',axis=1)
#td=td.drop('Ticket',axis=1)
#td=td.drop('Cabin',axis=1)
#td=td.drop('PassengerId',axis=1)
#print (td['Embarked'])
cols=['Sex']

x=preprocessing.LabelEncoder()


for i in td[cols].columns:
	td[i]=x.fit_transform(td[i])
	print (td[i])
mappu={'S':0,'Q':1,'C':2}
td['Embarked']=td['Embarked'].map(mappu)
print(td['Embarked']
#print (td)

#td['Sex']=x.fit_transform(td[Sex])
#td['Ticket']=x.fit_transform(td[Ticket])
#td['Cabin']=x.fit_transform(td[Cabin])
#td['Embarked']=x1.fit_transform(td[Embarked])

	
"""X_train=td[[x for x in td.columns if 'class' not in x]]
#print (X_train)
Y_train=td['class']
#print (Y_train)
y=preprocessing.LabelEncoder()
for col in td1.columns:
	td1[col]=y.fit_transform(td1[col])
X_test=td1[[x for x in td.columns if 'class' not in x]]
#print (X_test)
Y_test=td1['class']
#print (Y_test)
#from sklearn import linear_model
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'rbf',C=1).fit(X_train, Y_train)
#svm_model_linear = SVC(kernel = 'linear',C=1).fit(X_train, Y_train)
#svm_model_linear = linear_model.LogisticRegression().fit(X_train, Y_train)
#print (svm_model_linear)
#print (svm_model_linear.coef_)
svm_predictions = svm_model_linear.predict(X_test)
#print (svm_predictions)"""

 
"""# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, Y_test)
print (accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(Y_test, svm_predictions)
#print (cm)
#print (classification_report(Y_test,svm_predictions))"""


 
