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
#from sklearn import linear_model
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'rbf',C=1).fit(X_train, Y_train)
#svm_model_linear = linear_model.SGDClassifier().fit(X_train, Y_train)
#print (svm_model_linear)
#print (svm_model_linear.coef_)
svm_predictions = svm_model_linear.predict(X_test)
#print (svm_predictions)

 
# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, Y_test)
print (accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(Y_test, svm_predictions)
#print (cm)
print (classification_report(Y_test,svm_predictions))


 
