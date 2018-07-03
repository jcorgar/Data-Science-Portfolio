Download data from: http://www.geolink.pt/ecmlpkdd2015-challenge/dataset.html 

See ppt presentation for more details and data exploration.

Problem: Predict the amount of taxi rides on a given hour of a given day

Approach: Timestamps are first converted into date-time format and used as an index. Since each entry corresponds to one trip, a value of 1 is assigned to each row and then the data is interpolated by hour, yielding a column of total trips per hour. The rest of the columns can then be dropped as the only important features are the time and the number of trips per hour.

Models trained:

Holt Winters: https://www.otexts.org/fpp/7/5
SARIMA: https://www.otexts.org/fpp/8/9

Linear Regression:
A time series at time t depends on the previous steps of the series. For this reason, the features constructed for the regression were n steps (n from 6 ro 24). A lag of 24 was expected to be the most significant due to the 24 hour cycle. Other features, scalers, lasso and one hot encoding were tested but the simple model outperformed the more complicated models in accuracy.

Random Forest:
Same features as Linear regression. 

The models were compared against the heterogeneous ensemble method developed in: https://ieeexplore.ieee.org/document/6532415/




