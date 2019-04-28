import svm
import make_granules
import read_data
import numpy as np
def main():
	data = []
	inp = []
	num_iter = int ( input("Enter the number of iterations") )
	for i in range(0,num_iter):
		inp = make_granules.make_granules(i,data[:2])
		data = svm.svm(inp)

	print("clf is ",data[-1] )

	print("***********************************************************************")
	print("ON THE FINAL DATA !!!!")
	df = read_data.read_data()
	number_of_cols = len(df.columns)
	

	X = df.values
	Y = X[:,-1]
	X = X[:,:-1]

	indices = np.argwhere(Y==1)
	clf = data[-1]
	indices = indices.ravel()
	X = X[indices]

	predictions = clf.predict(X)
	correctly_done = np.sum(predictions)
	print("Correctly  classified minority points ",correctly_done/len(indices))
	print("***********************************************************************")

if __name__ == "__main__":
	main()