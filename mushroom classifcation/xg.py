from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing
import pandas as pd
import numpy as np

train=pd.read_csv("mushrooms.csv")
td=pd.DataFrame(train)
#print (td)
#print (train)
test=pd.read_csv("testmushroom.csv")
td1=pd.DataFrame(test)
#print (test)


x=preprocessing.LabelEncoder()
for col in td.columns:
	td[col]=x.fit_transform(td[col])
	#print (td[col])
X_train=td[[x for x in td.columns if 'class' not in x]]
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


#import xgboost as xgb
from xgboost import XGBClassifier
xgb_data=XGBClassifier().fit(X_train,Y_train)	
#print (xgb_data)

xgb_predictions = xgb_data.predict(X_test)
#print (xgb_predictions)

 
# model accuracy for X_test  
acc_train=xgb_data.score(X_train, Y_train)
#print (acc_train)
accuracy = xgb_data.score(X_test, Y_test)
print (accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(Y_test, xgb_predictions)
#print (cm)
print (classification_report(Y_test,xgb_predictions))


 
