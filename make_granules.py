'''
Input: Dataframe of the form [X Y]
where X consists of all the input features
and Y consists of all the label values
Output: 2 Dictionaries
granules contains the granules present within the dataset
labels contains the labels for each granule present within the dataset
'''
from sklearn import tree
import read_data
import pandas as pd
import numpy as np
from scipy import stats

def main():
	make_granules()

def make_granules(i,data):
	if i == 0:
		df = read_data.read_data()
		number_of_cols = len(df.columns)
		

		X = df.values
		Y = X[:,-1]
		X = X[:,:-1]

	else:
		X = data[0]
		Y = data[1]

	clf = tree.DecisionTreeClassifier()

	# print("Shape  of the input is")
	# print(X.shape)
	# print(Y.shape)
	Y = Y.astype('int')
	X = X.astype('float32')
	answer = clf.fit(X,Y) # end index is inclusive as it is hashed
   
	leaf_list = answer.tree_.apply(X) # excluding the last col

	granules = {} # dict to represent the granule
	labels = {} # dict to represent the label for a granule
	## Node each granule has a unique label

	index = 0

	# print("Got the nodes")
	# print(len(leaf_list))
	# print("Creating the dict")
	for leaf_index in leaf_list:
		if(leaf_index not in granules.keys()):
			granules[leaf_index] = X[index] # init first granule
			labels[leaf_index] = answer.predict( np.reshape(X[index],(1,X[index].shape[0]) ) )[0] # 1 row n cols
		else:
			granules[leaf_index] = np.vstack( (granules[leaf_index],X[index]) ) # make a ndarray of all records in a granule
		
		index += 1



	info_granules=None

	# changes made here onwards
	for key in granules:
		granule=granules[key]
		if len(granule.shape) == 1:
			granule = granule[np.newaxis,:]
		
		if info_granules is None:
			info_granules = stats.mode( np.asarray( granule ) )[0][0]
		else:
			info_granules = np.vstack( (info_granules,stats.mode( np.asarray( granule ) )[0][0] ) )
			
	   

	# print('Formed information granules:')
	# print(len(info_granules))
	# print(len(labels))
	labels = list(labels.values())
	labels = np.asarray(labels)
	# labels = np.reshape(labels,(labels.shape[0],1) )
	return [info_granules, labels]
		

if __name__ == '__main__':
	main()