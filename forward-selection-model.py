# a foward selection model for our dataset

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
# between this and 'GCSEgrade' which is what we already know!
drop_nan = data.drop('ATT8SCR', axis=1)
data = drop_nan

print(data.head())

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

selected_columns = my_fwd_selector(x_train, y_train, x_val, y_val)
print(selected_columns)
model = LogisticRegression(C=1e9).fit(x_train[selected_columns], y_train)

# these are simply arrays of 1s and 0s which is the model's predictions
y_train_predicted = model.predict(x_train[selected_columns])
y_val_predicted = model.predict(x_val[selected_columns])
y_test_predicted = model.predict(x_test[selected_columns])

# note that val accuracy is the accuracy when training the model and
# test accuracy is the accuracy on testing data set
print('======= Accuracy  table =======')
print('Training accuracy is:    {}'.format(accuracy_score(y_train, y_train_predicted)))
print('Validation accuracy is:  {}'.format(accuracy_score(y_val, y_val_predicted)))

'''

Finding the precision and recall.

'''
