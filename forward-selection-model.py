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

'''

The forward selector function.

'''

def my_fwd_selector(X_train, y_train, X_val, y_val):
    print('=============== Begining forward selection =================')
    cols = list(X_train.columns) # lists all the headers for each column in X_train (ie the variables)
    best_val_acc = 0
    selected_vars = []
    while len(cols) > 0: # 
        print('Trying {} var models'.format(len(selected_vars) + 1))
        candidate = None
        for i in range(len(cols)): # loops through all the variables
            current_vars = selected_vars.copy() # creates a shallow copy of selected_vars
            current_vars.append(cols[i]) # adds the current variable we are testing to current_vars
            if len(current_vars) == 1:
                new_X_train = X_train[current_vars].values.reshape(-1, 1) # convert data to a list of lists as sklearn expects
                new_X_val = X_val[current_vars].values.reshape(-1, 1)
            else:
                new_X_train = X_train[current_vars] # why doesn't it reshape here?                
                new_X_val = X_val[current_vars]
            
            mod = LogisticRegression(C=1e9).fit(new_X_train, y_train) # note that C is the inverse of regularization strength
            val_acc = accuracy_score(y_val, mod.predict(new_X_val))
            if val_acc - best_val_acc > 0.005: # if accuracy has improved...
                candidate = cols[i] # candidate is the successful variable (ie the one that has been added to the model)
                best_val_acc = val_acc
        if candidate is not None:
            selected_vars.append(candidate) # adds candidate to our list of variables
            cols.remove(candidate) # removes from cols so cannot be selected again
            print('------- Adding {} to the model ---------'.format(candidate))
        else:
            print('------- Accuracy not improved by adding another var -------')
            break
        print('Columns in current model: {}'.format(', '.join(selected_vars)))
        print('Best validation accuracy is {}'.format(np.round(best_val_acc, 3)))
    return selected_vars

'''

Training our model.

'''

#selected_columns = my_fwd_selector(X_train, y_train, X_val, y_val) # find the best variables to use in our model 

#model = LogisticRegression(C=1e9).fit(X_train[selected_columns], y_train) # train the model

'''

Finding the precision and recall.

'''
