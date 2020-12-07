import copy
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import confusion_matrix
from math import e


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

'''

Produce our model and find the accuracy, precision and recall.

'''

columns = ['P8MEA_17', 'KS2APS', 'PTFSM6CLA1A']
#columns = ['ATT8SCR_17', 'TOTPUPS', 'PTEBACMAT_E_PTQ_EE']
#columns = ['KS2APS', 'TOTPUPS']
#columns = ['PTEBACHUM_E_PTQ_EE', 'P8MEA_17', 'PTEBACLAN_E_PTQ_EE']
mod = LogisticRegression(C=1e9).fit(x_train[columns], y_train)

print('Attributes being tested are:     {}'.format(columns))

# TESTING SET
y_test_predicted = mod.predict(x_test[columns])
recall_test = recall_score(y_test, y_test_predicted, pos_label='1') # have to add pos_label so python knows which one is 'success'
precision_test = precision_score(y_test, y_test_predicted, pos_label='1')
accuracy_test = accuracy_score(y_test, y_test_predicted)
print('=============== Results from the TESTING set =================')
print('Recall is:     {}'.format(recall_test))
print('Precision is:  {}'.format(precision_test))
print('Accuracy is:  {}'.format(accuracy_test))


# VALIDATION SET
y_validation_predicted  = mod.predict(x_val[columns])
recall_validation = recall_score(y_val, y_validation_predicted, pos_label='1')
precision_validation = precision_score(y_val, y_validation_predicted, pos_label='1')
accuracy_validation = accuracy_score(y_val, y_validation_predicted)
print('=============== Results from the VALIDATION set =================')
print('Recall is:     {}'.format(recall_validation))
print('Precision is:  {}'.format(precision_validation))
print('Accuracy is:  {}'.format(accuracy_validation))

'''

Attempting to plot a LogReg graph for LogReg model 2 using test data.

(Not included in final report!)

# converting elements in y to ints
y_test_predicted = list(map(int, y_test_predicted))

# plotting all three attributes as a scatter
total_x = np.hstack(( x_test['P8MEA_17'], (x_test['KS2APS']/30), x_test['PTFSM6CLA1A'] )).ravel()
total_y = np.hstack(( y_test_predicted, y_test_predicted, y_test_predicted )).ravel()
plt.scatter(total_x, total_y, color='black',)

y_model_predicted = []
for i in range(len(x_test['P8MEA_17'])):
    y_model_predict_val = int(1 / (1 + ( e ** -( mod.coef_[0][0] * np.array(x_test['P8MEA_17'])[i] + mod.coef_[0][1] * np.array(x_test['KS2APS'])[i] + mod.coef_[0][2] * np.array(x_test['PTFSM6CLA1A'])[i])) ))
    y_model_predicted.append(y_model_predict_val)

# how do we combine all the attributes for the scatter plot?
x_combination = x_test['P8MEA_17'] + (x_test['KS2APS']/30) + x_test['PTFSM6CLA1A']
plt.scatter(x_combination, y_model_predicted)

# attempting to draw a log reg curve of best fit
#loss = expit(total_x).ravel()
plt.plot(total_x, loss, color='red', linewidth=3)

plt.xlabel('Attributes in model')
plt.ylabel('Predicted outcome')
plt.show()

'''


'''

Confusion matrix for the validation set.

'''

cm = confusion_matrix( y_val, y_validation_predicted)

plt.figure(figsize=(1,1))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["0", "1"], rotation=45, size = 15)
plt.yticks(tick_marks, ["0", "1"], size = 15)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')

plt.show()