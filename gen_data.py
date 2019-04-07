import numpy as np
import matplotlib.pyplot as plt


def gen_data(num_maj_class,num_min_class):

    # Hard coded params 
    mean = (0,0)
    cov = [[1,0],[0,1]]
    lower_lim = 0.9
    upper_lim = 2

    x , y = x, y = np.random.multivariate_normal(mean, cov, num_min_class).T
    


   

def main():
    num_maj_class = int("Enter the number of majority class points")
    num_min_class = int("Enter the number of  minority class points")
    gen_data(num_maj_class,num_min_class)

if __name__ == "__main__":
    main()