import pandas as pd
import numpy as np
def read_data():
    # Constants used
    file_path = 'mammographic_masses.data'
    missing_values = ["?"]

    df = pd.read_csv(file_path,header=None,na_values = missing_values)

    # Sanity check
    print("The Data Head is ")
    print(df.head())
    print("The summary of nulls before preprocessing")
    print(df.isnull().sum())

    df.fillna(method='ffill',inplace=True)
    print("After updation")
    print(df.head())

    ''' binning the age data to get ordinal attribute instead'''
    agebin = pd.cut(df.loc[:,1],bins=5,labels=range(1, 6), retbins=False,include_lowest=True)
    df.loc[:,1]=agebin
    print(df.head())


def main():
    read_data()

    

if __name__ == '__main__':
    main()

