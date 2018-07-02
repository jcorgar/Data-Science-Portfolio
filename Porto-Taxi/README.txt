Download data from: http://www.geolink.pt/ecmlpkdd2015-challenge/dataset.html 

Problem: Predict the amount of taxi rides on a given hour of a given day

Approach: Timestamps are first converted into date-time format and used as an index. Since each entry corresponds to one trip, a value of 1 is assigned to each row and then the data is interpolated by hour, yielding a column of total trips per hour. The rest of the columns can then be dropped as the only important features are the time and the number of trips per hour.




