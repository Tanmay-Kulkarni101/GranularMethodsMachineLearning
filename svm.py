'''
Input: Granules processed into representative elements X
where X is an ndarray of the representative datapoints
Y which are the labels for each granule
'''

from sklearn.svm import SVC
import numpy as np
import make_granules

def main():
	data = make_granules.make_granules(0,None) # Take data from read data
	svm(data)

def svm(data):
	## Add definitions for X and Y here
	X = data[0]
	Y = data[1]

	print(" We started with ",X.shape[0])
	clf = SVC() # using default arguments for SVM
	# C = 1 scaling for the error term
	# Kernel rbf 


	clf.fit(X,Y)

	results = clf.decision_function(X)
	indices = []
	for index in clf.support_ :
		if Y[index] == 0:
			indices.append(index)
	# print("indices are")
	# print(indices)
	# print("values are")
	# print(Y[indices])
	# print("X has ",X.shape[0] )
	A = np.delete(X,indices,axis = 0)
	B = np.delete(Y,indices)
	# print("Now we have ",A.shape[0])

	print(clf.score(X,Y))

	return [A,B,clf]

if __name__ == "__main__":
		main()