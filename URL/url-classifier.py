#Import libraries, and custom functions and classes
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np                             
import pandas as pd

#------------Functions----------------
def split_Ar(a,n_splits=10):
    """Split a sparse matrix into n_splits of equal length"""
	i1=0
	i2= a.shape[0]/n_splits
	s=[]
	while i2 <= a.shape[0]:
		s.append(a[i1:i2])
		i1=i2
		i2=i2+a.shape[0]/n_splits
	return s
    
def get_Accuracy(model,test_features, test_labels):
    """Returns the Accuracy of a model"""
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    return accuracy
#------------------Code----------------

print "Loading libraries complete"
#Load data
df= load_svmlight_file('Day0.svm')

print "Loading data complete"
#Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(df[0], df[1],
                                                    test_size = 0.20,
                                                    random_state = 42)
#Split traning data into smaller chunks for ease of computing
n_splits=10
Xs = split_Ar(X_train, n_splits)
ys = split_Ar(y_train, n_splits)

print "Start Training...\n"

#Initialize model with warm start to increase number of trees sequentially
rf = RandomForestClassifier(n_estimators = 10, random_state = 42,
                            warm_start=True)

#Iterate through chunks of training data and add a forest of 10 trees each time
for n in range(n_splits):
    print "Training model number",n+1
    rf.fit(Xs[n], ys[n])
    rf.n_estimators += 10



print "Final model has accuracy of %.2f\n" % (get_Accuracy(rf,X_test,y_test))


