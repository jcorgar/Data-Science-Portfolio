Each data sample corresponds to one completed trip. It contains a total of 9 (nine) features, described as follows:

TRIP_ID: (String) It contains a unique identifier for each trip;

CALL_TYPE: (char) It identifies the way used to demand this service. It may contain one of three possible values:
- 'A' if this trip was dispatched from the central;
- 'B' if this trip was demanded directly to a taxi driver at a specific stand;
- 'C' otherwise (i.e. a trip demanded on a random street).

ORIGIN_CALL: (integer) It contains a unique identifier for each phone number which was used to demand, at least, one service. It identifies the trip's customer if CALL_TYPE='A'. Otherwise, it assumes a NULL value;

ORIGIN_STAND: (integer): It contains a unique identifier for the taxi stand. It identifies the starting point of the trip if CALL_TYPE='B'. Otherwise, it assumes a NULL value;

TAXI_ID: (integer): It contains a unique identifier for the taxi driver that performed each trip;

TIMESTAMP: (integer) Unix Timestamp (in seconds). It identifies the trip's start;

DAYTYPE: (char) It identifies the daytype of the trip's start. It assumes one of three possible values:
- 'B' if this trip started on a holiday or any other special day (i.e. extending holidays, floating holidays, etc.);
- 'C' if the trip started on a day before a type-B day;
- 'A' otherwise (i.e. a normal day, workday or weekend).

IMPORTANT NOTICE: This field has not been correctly calculated. Please see the following links as reliable sources for official holidays in Portugal.
[Web Link]
[Web Link]

MISSING_DATA: (Boolean) It is FALSE when the GPS data stream is complete and TRUE whenever one (or more) locations are missing;

POLYLINE: (String): It contains a list of GPS coordinates (i.e. WGS84 format) mapped as a string. The beginning and the end of the string are identified with brackets (i.e. [ and ], respectively). Each pair of coordinates is also identified by the same brackets as [LONGITUDE, LATITUDE]. This list contains one pair of coordinates for each 15 seconds of trip. The last list item corresponds to the trip's destination while the first one represents its start.


Relevant Papers:

Moreira-Matias L., Gama J., Ferreira M., Mendes-Moreira J. and Damas L.,: "Time-Evolving OD Matrix Estimation using high-speed GPS data streams". In: Expert Systems with Applications, vol. 44, pp. 275-288, February (2016)

Moreira-Matias, L., Gama, J., Ferreira, M., Mendes-Moreira, J., Damas, L., ”Predicting Taxi–Passenger Demand Using Streaming Data”. In: IEEE Transactions on Intelligent Transportation Systems, vol.14, no.3, pp.1393-1402, September (2013)



Citation Request:

Moreira-Matias, L., Gama, J., Ferreira, M., Mendes-Moreira, J., Damas, L., ”Predicting Taxi–Passenger Demand Using Streaming Data”. In: IEEE Transactions on Intelligent Transportation Systems, vol.14, no.3, pp.1393-1402, September (2013)
