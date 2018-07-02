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

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski').fit(X_train, Y_train)

#print (knn)

knn_predictions = knn.predict(X_test)
#print (knn_predictions)

 
# model accuracy for X_test  
accuracy = knn.score(X_test, Y_test)
print (accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(Y_test, knn_predictions)
#print (cm)
print (classification_report(Y_test,knn_predictions))


 
