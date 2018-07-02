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
#from sklearn import linear_model
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'rbf',C=1).fit(X_train, Y_train)
#svm_model_linear = SVC(kernel = 'linear',C=1).fit(X_train, Y_train)
#svm_model_linear = linear_model.LogisticRegression().fit(X_train, Y_train)
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
#print (classification_report(Y_test,svm_predictions))


 
