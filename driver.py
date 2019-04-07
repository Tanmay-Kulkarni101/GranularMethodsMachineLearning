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

    granules = {}
    labels = {}

    index = 0

    print("Got the nodes")
    print(len(leaf_list))
    print("Creating the dict")
    for leaf_index in leaf_list:
        if(leaf_index not in granules.keys()):
            granules[leaf_index] = X[index]
            labels[leaf_index] = answer.predict( np.reshape(X[index],(1,X[index].shape[0]) ) )
        else:
            granules[leaf_index] = np.vstack( (granules[leaf_index],X[index]) )
        
        index += 1

    print("Added the value")
    print(granules)


if __name__ == '__main__':
    main()