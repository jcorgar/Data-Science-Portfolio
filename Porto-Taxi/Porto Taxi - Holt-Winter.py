#Import libraries, and custom functions and classes
from HelperHW import *

#------------------Code----------------

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
data = df.Trips[:720] # leave some data for testing

# initializing model parameters alpha, beta and gamma
x = [0, 0, 0] 

# Minimizing the loss function 
opt = minimize(timeseriesCVscore, x0=x, 
               args=(data, mean_squared_error), 
               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
              )
print "Optimization complete"
# Take optimal values...
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)

# ...and train the model with them, forecasting for the next 50 hours
model = HoltWinters(data, slen = 24, 
                    alpha = alpha_final, 
                    beta = beta_final, 
                    gamma = gamma_final, 
                    n_preds = 50, scaling_factor = 3)
model.triple_exponential_smoothing()
print "Model complete"
plotHoltWinters(df.Trips[:720],model)
plt.show()
#plotHoltWinters(df.Trips[:720],model, plot_intervals=True, plot_anomalies=True)
'''forecast=model.result[:720+24]
mean_absolute_percentage_error(df.Trips[720:744],forecast[720:])'''
