from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
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


from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, Y_train)
#dtree_model = RandomForestClassifier(max_depth = 2).fit(X_train, Y_train)
#print (dtree_model)
#dtree_predictions = dtree_model.predict_proba(X_test)
dtree_predictions = dtree_model.predict(X_test)
#print (dtree_predictions)

# creating a confusion matrix
cm = confusion_matrix(Y_test, dtree_predictions)
#print (cm)
accuracy=dtree_model.score(X_test,Y_test)
print (accuracy)
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
