#Import libraries, and custom functions and classes
from Helper import *

#------------------Code----------------
print "Finished loading libraries"
# Read only the columns corresponding to the Timestamp
df= pd.read_csv('train.csv', usecols=[5])
#Since each row is one taxi trip, create column of 1's
df['Trips']=pd.Series(1,index=df.index)

#Make the Timestamp into and actual datetime object and set as index
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'],unit='s').sort_values()
df.set_index('TIMESTAMP', inplace=True)
#Add up the number of trip per hour
df= df.resample('H').sum()

#Ranges for the p,q, P, Q hyperparameters
ps = range(1, 3)
d=1 
qs = range(1, 3)
Ps = range(0, 3)
D=1 
Qs = range(0, 2)
s = 24

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print len(parameters_list)

print "Starting fits..."



#Creates a grid of SARIMA parameters and return the best combination
#The most time consuming step of the program
#result_table = optimizeSARIMA(df.Trips[:len(df.Trips)/2],parameters_list, d, D, s)

#p, q, P, Q = result_table.parameters[0]
#Best hyperparameters for SARIMA
p,q,P,Q=(2, 2, 0, 1)
#Best paramters for SARIMA
parametros= [.3127, -.4453, .2061, .5943, -.9956, 929.4983]
#best_model=sm.tsa.statespace.SARIMAX(ads.Ads, order=(p, d, q), 
#                seasonal_order=(P, D, Q, s)).fit(disp=-1)

best_model=sm.tsa.statespace.SARIMAX(ads.Ads, order=(p, d, q), 
                seasonal_order=(P, D, Q, s)).fit(start_params=parametros,disp=-1)
