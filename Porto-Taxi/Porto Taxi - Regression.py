#Import libraries, and custom functions and classes
from HelperLR import *

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

#Creat and add lag features
for i in range(6, 25):
    data["lag_{}".format(i)] = data.y.shift(i)


y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

# Split set
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0)

# Regression fitting
lr = LinearRegression()
lr.fit(X_train, y_train)


'''
plotModelResults(lr,X_train,X_test, y_train, y_test, plot_intervals=True)
plt.show()
plotCoefficients(lr,X_train)
plt.show()


data["hour"] = data.index.hour
data["weekday"] = data.index.weekday
data['is_weekend'] = data.weekday.isin([5,6])*1

scaler = StandardScaler()

y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_R = LinearRegression()
lr_R.fit(X_train_scaled, y_train)

#plotModelResults(lr, X_train_scaled, X_test_scaled, y_train, y_test, plot_intervals=True)
#plotCoefficients(lr,X_train)
'''
