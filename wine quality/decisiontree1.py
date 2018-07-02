from sklearn.metrics import confusion_matrix,classification_report
#from sklearn.model_selection import train_test_split

#import graphviz
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
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, Y_train)
dtree_model = RandomForestClassifier(max_depth = 2).fit(X_train, Y_train)
#print (dtree_model)
#dtree_predictions = dtree_model.predict_proba(X_test)
dtree_predictions = dtree_model.predict(X_test)
#print (dtree_predictions)

# creating a confusion matrix
cm = confusion_matrix(Y_test, dtree_predictions)
#print (cm)
accuracy=dtree_model.score(X_test,Y_test)
#print (accuracy)
ncm= cm/cm.astype(np.float).sum(axis=1)
#print (ncm)
print (classification_report(Y_test,dtree_predictions))


"""dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=X_train,
                         class_names=Y_train,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("train")"""
