# %%
import pandas as pd 
import numpy as np 

housing = pd.read_csv("datasets/housing/housing.csv")

# %%
housing

# %%
def split_train_test(data , test_ratio):
    np.random.seed(42)
    shuffled_indeces = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indeces = shuffled_indeces[:test_set_size]
    train_indeces = shuffled_indeces[test_set_size:]

    return data.iloc[test_indeces] , data.iloc[train_indeces]

    
random_test_set , random_train_set = split_train_test(housing , 0.2)


# %%
housing["income_cat"] = pd.cut(housing["median_income"] , bins= [0,1.5,3,4.5,6,np.inf] , labels= [1,2,3,4,5] )

housing["income_cat"].hist()

# %%
random_train_set.reset_index()

random_train_set["income_cat"] =  pd.cut(random_train_set["median_income"] , bins= [0,1.5,3,4.5,6,np.inf] , labels= [1,2,3,4,5])

random_train_set_bin_distribution =random_train_set["income_cat"].value_counts()/len(random_train_set)

# %%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state= 42) 

for train_index , test_index in split.split(housing , housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


strat_train_set_bin_distribution= strat_train_set["income_cat"].value_counts()/len(strat_train_set)


# %%
def percentage_error(initial , final ):
    return ((final - initial)/final)*100 

# %%
overall_bin_distribution = housing["income_cat"].value_counts()/len(housing)

overall = pd.DataFrame(overall_bin_distribution)


# %%
dict1 = {
    'Overall' : overall_bin_distribution,
    'Random' : random_train_set_bin_distribution,
    'Stratified' : strat_train_set_bin_distribution,
    'Strat.%error' : percentage_error(strat_train_set_bin_distribution, overall_bin_distribution),
    'Ramdom %error' : percentage_error(random_train_set_bin_distribution, overall_bin_distribution)
}

compare = pd.DataFrame(dict1)

# %%
compare =compare.sort_values(by = ['income_cat'])

# %%


# %%


# %%


# %%
import pickle

model_loaded = pickle.load(open('models/RandomForest', 'rb'))

# %%


# %%



