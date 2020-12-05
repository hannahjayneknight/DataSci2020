import copy
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('LondonGCSEData.csv') # import csv data here

'''
Here we binarize dependent variable. This is done by creating a column with 1s and 0s
where 1 = 'good' grade and 0 = 'bad' grade. 
How was this calculated?
Firstly, the mean attainment 8 score for all schools was found. Schools with an attainment 
8 higher than the mean got a 'good' score.
'''

meangrade = data['ATT8SCR'].mean()
sdgrade = data['ATT8SCR'].std()
data['GCSEgrade'] = np.where(data['ATT8SCR'] >= meangrade , '1', '0')

# get rid of the original 'ATT8SCR' column so that we do not get a linear relationship
# between this and 'GCSEgrade' which is what we already know! Also drop school identifier
drop_nan = data.drop('ATT8SCR', axis=1)
data = drop_nan
drop_nan = data.drop('URN', axis=1)
data = drop_nan
drop_nan = data.drop('P8MEA', axis=1)
data = drop_nan


'''
Split the data and separate into respective X and y dataframes.
'''

train, other = train_test_split(data, test_size=0.2, random_state=0)
validation, test = train_test_split(other, test_size=0.5, random_state=0)

x_train = train.drop(columns=['GCSEgrade'])
y_train = train['GCSEgrade']

x_val = validation.drop(columns=['GCSEgrade'])
y_val = validation['GCSEgrade']

x_test = test.drop(columns=['GCSEgrade'])
y_test = test['GCSEgrade']


def my_fwd_selector(X_train, y_train, X_val, y_val, cols):
    print('=============== Begining forward selection =================')
    print('Starting with {} as the first attribute'.format(cols[0]))
    cols_copy = copy.deepcopy(cols)
    best_val_acc = 0
    selected_vars = []
    while len(cols_copy) > 0:

        print('Trying {} var model'.format(len(selected_vars) + 1))
        candidate = None

        for i in range(len(cols_copy)):
            current_vars = selected_vars.copy() 
            current_vars.append(cols_copy[i]) 
            if len(current_vars) == 1:
                new_X_train = X_train[current_vars].values.reshape(-1, 1) # convert data to a list of lists as sklearn expects
                new_X_val = X_val[current_vars].values.reshape(-1, 1)
            else:
                new_X_train = X_train[current_vars] # why doesn't it reshape here?                
                new_X_val = X_val[current_vars]
            
            mod = LogisticRegression(C=1e9).fit(new_X_train, y_train) # note that C is the inverse of regularization strength
            val_acc = accuracy_score(y_val, mod.predict(new_X_val))
            if val_acc - best_val_acc > 0.005: # if accuracy has improved...
                candidate = cols_copy[i] # candidate is the successful variable (ie the one that has been added to the model)
                best_val_acc = val_acc

        if candidate is not None:
            selected_vars.append(candidate) # adds candidate to our list of variables
            cols_copy.remove(candidate) # removes from cols_copy so cannot be selected again
            print('------- Adding {} to the model ---------'.format(candidate))
        else:
            print('------- Accuracy not improved by adding another var -------')
            break
        print('Columns in current model: {}'.format(', '.join(selected_vars)))
        print('Best validation accuracy is {}'.format(np.round(best_val_acc, 3)))
    return selected_vars

'''

This forward selector model loops through every attribute and
uses the previous forward selector function on each one as the
starting attribute has a big effect on the final model.

'''

def fwd_select_every_attribute(X_train, y_train, X_val, y_val):
    cols = list(X_train.columns) # lists all the attributes

    # for each attribute, makes a new array with the attribute being added first
    # at the start and the remaining attributes following
    for i in range(len(cols)):
        if i == 0:
            my_fwd_selector(X_train, y_train, X_val, y_val, cols)
            #print(cols)
            #print(cols)
            
        else:
            new_cols = copy.deepcopy(cols)
            var = new_cols[i]
            new_cols.remove(var)
            new_cols.insert(0, var)
            my_fwd_selector(X_train, y_train, X_val, y_val, new_cols)
            #print(new_cols)

    return 


print(fwd_select_every_attribute(x_train, y_train, x_val, y_val))
