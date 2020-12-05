'''

This program aims to find the best Logression Model for a dataset by
testing every possible combination of attributes in the dataset. To 
cope with this huge amount of data, it uses the "multiprocessing" 
package to use all the available processors on the computer it is
being run on.

'''

from itertools import combinations
import pandas as pd
import copy
import warnings
warnings.filterwarnings("ignore")

'''

This function lists EVERY possible combination of variables for the model

'''

data = pd.read_csv('LondonGCSEData.csv') # import csv data here

def our_models_to_test(attributes):
    #group_size = len(attributes)
    models = []
    for x in range(len(attributes)):
        comb = list(combinations(attributes, x))
        models.append(comb)

    return models

#attributes = data.columns.values.tolist()
#print(our_models_to_test(attributes))