# a foward selection model for our dataset

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('england_ks4final.csv') # import csv data here
print(data.head())
#train, other = train_test_split(data, test_size=0.2, random_state=0)
#validation, test = train_test_split(other, test_size=0.5, random_state=0)


