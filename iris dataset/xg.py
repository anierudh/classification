from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd
import numpy as np
train=pd.read_csv("Iris.csv")
#print (train)
test=pd.read_csv("iristest.csv")
#print (test)
features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X_train=train[list(features)].values
#print (X_train)
Y_train=train['Species'].values
#print (Y_train)
X_test=test[list(features)].values
#print (X_test)
Y_test=test['Species'].values
#print (Y_test)

from xgboost import XGBClassifier
xgb_data=XGBClassifier().fit(X_train,Y_train)	
#print (xgb_data)

xgb_predictions = xgb_data.predict(X_test)
#print (xgb_predictions)

 
# model accuracy for X_test  

accuracy = xgb_data.score(X_test, Y_test)
print (accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(Y_test, xgb_predictions)
#print (cm)
print (classification_report(Y_test,xgb_predictions))


 
