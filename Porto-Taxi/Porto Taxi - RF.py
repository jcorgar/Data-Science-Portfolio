#Import libraries, and custom functions and classes
from HelperRF import *

#------------------Code----------------
print "Loading libraries complete"
# Read only the columns corresponding to the Timestamp
df= pd.read_csv('train.csv', usecols=[5])
#Since each row is one taxi trip, create column of 1's
df['Trips']=pd.Series(1,index=df.index)

#Make the Timestamp into and actual datetime object and set as index
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'],unit='s').sort_values()
df.set_index('TIMESTAMP', inplace=True)
#Add up the number of trip per hour
df= df.resample('H').sum()

print "Loading data complete"

data = df[:720]
data.columns = ["y"]

#Creat lag features
for i in range(6, 25):
    data["lag_{}".format(i)] = data.y.shift(i)

#Add categorical features
data["hour"] = data.index.hour
data["weekday"] = data.index.weekday
data['is_weekend'] = data.weekday.isin([5,6])*1

scaler = StandardScaler()

y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)
#Scale features for regularization
X_scaled = scaler.fit_transform(X)
print "Start Training..."
#1000 trees with seed fixed at 42 for consistency
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_scaled, y)
