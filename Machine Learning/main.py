import pandas as pd 
import numpy as np

# housing = pd.read_csv("housing.csv")
# # housing.info()
# # print (housing["ocean_proximity"].value_counts())
# print(housing.describe() )

def split_train_data(data , test_ratio):
    
    shuffled_indces = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    train_indeces= shuffled_indces[:test_set_size]
    test_indeces = shuffled_indces[test_set_size:] 
    return data.iloc[train_indeces], data.iolc[test_indeces]