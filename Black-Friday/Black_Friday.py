#Import libraries, and custom functions and classes
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
import numpy as np                             
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#from feature_selector import FeatureSelector
#------------Functions----------------
def split_Ar(a,n_splits=10):
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

def sanitize(X):
        X["Gender"] = le.fit_transform(X["Gender"])
        X["Product_ID"] = le.fit_transform(X["Product_ID"])
        X["City_Category"] = le.fit_transform(X["City_Category"])
        X["Age"] = le.fit_transform(X["Age"])
        X["Stay_In_Current_City_Years"] = le.fit_transform(X["Stay_In_Current_City_Years"])


        X['Product_Category_1']= X['Product_Category_1'].cat.add_categories([0])
        X['Product_Category_2']= X['Product_Category_2'].cat.add_categories([0])
        X['Product_Category_3']= X['Product_Category_3'].cat.add_categories([0])

        X['Product_Category_1']=X['Product_Category_1'].fillna(0)
        X['Product_Category_2']=X['Product_Category_2'].fillna(0)
        X['Product_Category_3']=X['Product_Category_3'].fillna(0)
        return X
#------------------Code----------------

print "Loading libraries complete"
#Load data
df=pd.read_csv('train.csv',dtype='category')
y=df['Purchase'].astype(np.int32)

X = df.drop(['Purchase'], axis=1)
X = df.drop(['User_ID'], axis=1)

le = LabelEncoder()
#onehot_encoder = OneHotEncoder()

X = sanitize(X)

n_splits=1000
Xs = split_Ar(X, n_splits)
ys = split_Ar(y, n_splits)

print "Start Training"
#fs = FeatureSelector(data = train, labels = train_labels)

rf = RandomForestRegressor(n_estimators = 10, random_state = 42,
                            warm_start=True, oob_score=
                            True,max_features='sqrt')

for n in range(n_splits):
        #fs = FeatureSelector(data = X[n], labels = y[n])
        #train_removed = fs.remove(methods = 'all')
        rf.fit(Xs[n], ys[n])
        if n%100==0:
                print "Training model number",n+1
        rf.n_estimators += 10

                            


