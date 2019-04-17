'''
Input: Granules processed into representative elements X
where X is an ndarray of the representative datapoints
Y which are the labels for each granule
'''

from sklearn.svm import SVC

## Add definitions for X and Y here
# X =
# Y =
clf = SVC() # using default arguments for SVM
# C = 1 scaling for the error term
# Kernel rbf 

# boundry Heuristic? How close is close enough?
decision_heuristic = 2
clf.fit(X,Y)

results = clf.decision_function(X)

# should return those values that satisfy the heuristic
X = X[(results >  decision_heuristic) | (results < - decision_heuristic ) ]
Y = Y[(results >  decision_heuristic) | (results < - decision_heuristic ) ]
