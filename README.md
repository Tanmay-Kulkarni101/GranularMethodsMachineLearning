# GranularMethodsMachineLearning
Implement granular methods in machine learning. The purpose for using granular methods is to allow an ordianry classifier such as a support vector machine to deal with highly imbalenced datasets.

## Description of Dataset
### Labels
  - Benign=0
  - Malignant=1

### Features
  - BI-RADS assessment: 1 to 5 (ordinal)  
  - Age: patient's age in years (integer) 
  - Shape: mass shape: (nominal)
      - round=1 
      - oval=2 
      - lobular=3 
      - irregular=4 
  - Margin: mass margin: (nominal)
      - circumscribed=1 
      - microlobulated=2 
      - obscured=3 
      - ill-defined=4 
      - spiculated=5 
  - Density: mass density (ordinal)
      - high=1 
       - iso=2 
       - low=3 
       - fat-containing=4 
  - Severity: (binominal)
       - benign=0
       - malignant=1 

## Preprocessing
Eliminate null values with the help of forward fill
Create bins based on the age of the women, into 5 bins according to the data range

## Modules
- read_data : Has a function that reads data from a text file and preprocesses the data, which it returns to a pandas dataframe.
- make_granule: 
Runs decision tree on the data to make granules.
Choose the mode amongst all the points present in the granule
Return this data along with the corresponding labels
- svm(): 
 Run the svm algorithm on the data sent by make_granule and return the decision boundary and support vectors.
Remove those granules that lie on the decision boundary.
- driver(): the function that runs all the modules in order.

## Implementation
- Implement granularization with the help of decision trees.
- Choose the mode amongst the points present in the granule.
- Run svm on the granules.
- Remove those granules that lie on the support vector from the majority class.
- Repeat the first three steps till we hit the upper bound on the number of steps or minority class misclassification is 0.

![SVM Boundry Image](600px-SVM_margin.png "SVM")

## Dataset
( http://archive.ics.uci.edu/ml/datasets/mammographic+mass )

![Working Example](img.PNG "Working Example")

## CONCLUSION
The class imbalance problem is present within a majority of the datasets, as it is higly improbable to expect a balenced distribution of labels within a classificaiton problem. This problems cannot be solved by the Machine Learning Model alone, thus, we make use of specialized techniques that make the training models immune to these imbalances within the dataset. This technique focuses on creating granules of the input data and then runs training models, so as to prune borderline cases and improve the performance on the minority class.

This algorithm purposely under fits the data, so that we do not bias based on the majority class. Thus, we may have a lower accuracy overall, but a higher accuracy for the minority class. These decision bounderies tend to gerneralize better as they are not biased by the majority class.

![GSVM in Action](GSVM.gif "GSVM in action")
## Contact the Authors
- Tanmay Kulkarni (f20150647@hyderabad.bits-pilani.ac.in)
- Somya Sharma (f20160216@hyderabad.bits-pilani.ac.in)

## Citation
This work is based on the following paper:
He, Haibo, and Edwardo A. Garcia. "Learning from imbalanced data." IEEE Transactions on knowledge and data engineering 21.9 (2009): 1263-1284.
