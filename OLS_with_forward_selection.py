# uploading the csv to the google collaborator file
from google.colab import files
uploaded = files.upload()

# importing everything that is necessary
import statsmodels.formula.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import t as tdist
import warnings
warnings.filterwarnings('ignore')

# reading the csv and data types for each column
gcse_scores = pd.read_csv('LondonGCSEData1.csv')
gcse_scores.info(verbose = True)

# clean the code to get rid of the columns we dont want
gcse_scores = gcse_scores.drop('URN', axis=1) # URN is the number that describes a school which is not a variable
gcse_scores = gcse_scores.drop('SCHOOLTYPE', axis=1) # schooltype is London (4) for all schoolsi not an indicator and variable
gcse_scores.head()

def my_function(columns_in_model, columns_to_add):
#columns_to_add is the list of all the columns we want to add for feature selection process

  if len(columns_in_model) == 0: # if we start with no variables in the model

    list_of_trials = [] # empty list that adds the rsquares each time a feature is test
    already_in = [] # empty list that stores the features selected for the model

    for i in range(len(columns_to_add)): # creating a for loop that goes through all the feature candidates 
    
      my_column_name = columns_to_add[i] 
      #selects a feature each time
      #print(my_column_name) 'TOTPUPS' 'uncomment if you want to see which feature is being calculated

      initial_model = sm.ols('ATT8SCR ~ ' + str(my_column_name), gcse_scores).fit()
      # creating the ols model that specifies which is the dependent varibale :'ATT8SCR' 
      # and adds the selected feature into the model to fit the data

      list_of_trials.append(initial_model.rsquared) 
      # gets the rsquared value of the model and stores the values for every feature in a list

    print('Features {} are on trial for rsquare performance'.format(columns_to_add))
    print('Rsquared values are calculated as {}'.format(list_of_trials)) 
    # prints the list of features as it will decrease by removing the selected features 
    # prints the list of rsquared values

    index_for_max = list_of_trials.index(max(list_of_trials))
    # finds the index of the maximum value in the list of rsquared values

    column_addition = columns_to_add[index_for_max]
    # pfinds the name of the column that gave the highes rsquared and stores it as column_addition varibale
    print('Feature {} is added to the model because it has the highest rsquare value! '.format(column_addition))
    # prints the name of the best feature

    already_in = column_addition
    # adds the selected feature for the model in the list already_in that will contain every feature selected in the end
    columns_to_add.remove(column_addition)
    # removes the feature selected from the features on trial list to avoid trying it again
    
  
    for trials in range(3): 
    # selecting best 3 features that increase rsquare value- selecting top 4 in the end because we chose one before this loop as the initial one
    # going through the same process again where every feature is tried as the second feature to be added and the results are stored in a list
      list_of_trials = []

      for i in range(len(columns_to_add)):
        
        my_column_name = columns_to_add[i] #TOTPUPS

        #print(my_column_name) # TOTPUPS --- PTEBACLAN_E_PTQ_EE
        #print(str(already_in)) # PTEALGRP2 --- PTEALGRP2
        
        model = sm.ols('ATT8SCR ~ ' + str(already_in) + '+' + str(my_column_name), gcse_scores).fit()
        # selected features are the already_in that increases everytime new feature is selected (see below) and different column names are on trial

        list_of_trials.append(model.rsquared)

      print('Features {} are on trial for rsquare performance'.format(columns_to_add))
      print('Rsquared values are calculated as {}'.format(list_of_trials))
      
      index_for_max = list_of_trials.index(max(list_of_trials))
      
      column_addition = columns_to_add[index_for_max]
      
      print('Feature {} is added to the model because it has the highest rsquare value!'.format(column_addition))
      print('Rsquared value is {}'.format(list_of_trials[index_for_max]))

      already_in = already_in + ' + ' + column_addition
      # storing the selected features

      columns_to_add.remove(column_addition)
      # removing the selected features from the ones that are going to be tested
    
    print('Top 4 features selected for the OLS regression model using forward selection are: {} '.format(already_in))
    # printing the top 4 selected

my_function([], ['ATT8SCR_17', 'P8MEA', 'KS2APS', 'ADMPOL_PT', 'P8MEA_17', 'PSEN_ALL', 'PTEBACHUM_E_PTQ_EE'])
# trying the function with continuous data