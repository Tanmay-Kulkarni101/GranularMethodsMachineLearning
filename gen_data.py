import numpy as np
import matplotlib.pyplot as plt
import pickle

def gen_data(num_maj_class,num_min_class):

    # Hard coded params 
    mean = (0,0)
    cov = [[1,0],[0,1]]
    lower_lim = 0.9
    upper_lim = 2
    


    x , y  = np.random.multivariate_normal(mean, cov, num_min_class).T
    data = {}
    data_points = {}
    data_points['x'] = x
    data_points['y'] = y
    data['minor'] = data_points
    print("Minority data")
    print(data)


    temp = np.random.uniform(low = lower_lim, high=upper_lim , size = (num_maj_class,2) )
    x = temp[:,0]
    y = temp[:,-1]
    temp = np.random.uniform(low = -upper_lim, high=-lower_lim , size = (num_maj_class,2) )
    x = np.concatenate((x,temp[:,0]))
    y = np.concatenate((y,temp[:,1]))
    
    data = {}
    data_points = {}
    data_points['x'] = x
    data_points['y'] = y
    data['major'] = data_points
    print("Majority data")
    print(data)

    with open('data.p','wb') as file_handle:
        pickle.dump(data,file_handle)

   

def main():
    num_maj_class = int( input ("Enter the number of majority class points") )
    num_min_class = int( input("Enter the number of  minority class points") )
    gen_data(num_maj_class,num_min_class)

if __name__ == "__main__":
    main()