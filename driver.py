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
def main():
    df = read_data.read_data()
    number_of_cols = len(df.columns)
    clf = tree.DecisionTreeClassifier()

    X = df.values
    Y = X[:,-1]
    X = X[:,:-1]
    
    print("Shape  of the input is")
    print(X.shape)
    print(Y.shape)
    Y = Y.astype('int')
    X = X.astype('float32')
    answer = clf.fit(X,Y) # end index is inclusive as it is hashed
   
    leaf_list = answer.tree_.apply(X) # excluding the last col

    granules = {} # dict to represent the granule
    labels = {} # dict to represent the label for a granule
    ## Node each granule has a unique label

    index = 0

    print("Got the nodes")
    print(len(leaf_list))
    print("Creating the dict")
    for leaf_index in leaf_list:
        if(leaf_index not in granules.keys()):
            granules[leaf_index] = X[index] # init first granule
            labels[leaf_index] = answer.predict( np.reshape(X[index],(1,X[index].shape[0]) ) ) # 1 row n cols
        else:
            granules[leaf_index] = np.vstack( (granules[leaf_index],X[index]) ) # make a ndarray of all records in a granule
        
        index += 1

    print("Added the value")
    # print(granules)
    print(granules[251])


if __name__ == '__main__':
    main()