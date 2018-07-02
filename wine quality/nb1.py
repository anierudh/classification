from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
train=pd.read_csv("winequality-red.csv")
#print (train)
test=pd.read_csv("testwine.csv")
#print (test)
features=['citric acid','residual sugar','chlorides','density','pH','alcohol','fixed acidity','volatile acidity','free sulfur dioxide','total sulfur dioxide','sulphates']
X_train=train[list(features)].values
#print (X_train)
Y_train=train['quality'].values
#print (Y_train)
X_test=test[list(features)].values
#print (X_test)
Y_test=test['quality'].values
#print (Y_test)
from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
gnb = BernoulliNB().fit(X_train, Y_train)
#gnb = MultinomialNB().fit(X_train, Y_train)
#gnb = GaussianNB().fit(X_train, Y_train)
#print (gnb)
gnb_predictions = gnb.predict(X_test)
#gnb_predictions = gnb.predict_proba(X_test)
#print (gnb_predictions) 
# accuracy on X_test
accuracy = gnb.score(X_test, Y_test)
print (accuracy)
 
# creating a confusion matrix
cm = confusion_matrix(Y_test, gnb_predictions)
print (cm)
