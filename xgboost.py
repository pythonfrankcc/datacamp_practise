#using xtreme boost as an ensemble language
#lets start with reading the data
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
#importing the xgboost API for xgboost 
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)
#now lets make the first naive model from which our model learns
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
#now lets move to parameter tuning 
'''lets look at the parameters that you could tune and what they stand for:
1)n_estimators --specifies how many times to go through the modeling cycle. 
It is equal to the number of models that we include in the ensemble.
A large value would result in overfitting while a small value would result in underfitting but the typical valiue 
varies from 100-1000 with the value dependent on the learning rate'''
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)
'''2)early_stopping_rounds--early stopping round dictates after how many deteriorating iterations should it
stop when trying to find the ideal n_estimators.It is normally idael to set a high n_estimator and then set an 
early_stoopping_rounds to find the ideal n_estimators.
When using early_stopping_rounds its also important to set aside some data for calculating the validation scores
which is noemally done by setting the eval_set parameter.
'''
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
'''3)learning_rate--this helps reduce the fact that you have to add the previous model to the next by just
multiplying the previous model using a small no(learning_rate) this helps reduce the chances of overfitting 
even with a large no of the n_estimators ,i.e, best to use a small learning_rate and a high n_estimators
The default learning_rate of xgboost is 0.1'''
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
'''4)n_jobs-->This is particularly useful when you are building models using large datasets and you need to apply 
parallelism as n_jobs is equal to the number of cores in your machine but on a small dataset this won't help
This does not improve the accuracy but only reduces the time taken to do the fitting'''
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False) 
