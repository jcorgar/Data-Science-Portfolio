Dataset can be downloaded at: http://www.sysnet.ucsd.edu/projects/url/

The data is composed of 16k rows and ~3.5MM columns. 

Since the data cannot fit into memory, the Random Forest model was chosen because of how the voting system in the forest works. Random Forest uses bagging to decrease variance in the estimators without affecting the bias. This means that the "decision" comes down to a weighted sum between all of the trees in the forest. There are too many rows and features so using all of the data is impractical. The work-around I came up with is to split the training set into smaller parts (10 in this case) and train a small forest (10 trees) with those. If we aggregate all of the forests from the previous process, we will get a bigger forest that was trained in all of the training data. 

The final model functions as a forest trained on all of the data with 100 trees and has an accuracy of 96%. This process is easy enough to extend and do in parallel for bigger data-sets or for more trees.
