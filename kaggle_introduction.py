
# coding: utf-8

# **[Intermediate Machine Learning Micro-Course Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# ---
# 

# As a warm-up, you'll review some machine learning fundamentals and submit your initial results to a Kaggle competition.
# 
# # Setup
# 
# The questions below will give you feedback on your work. Run the following cell to set up the feedback system.

# In[2]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex1 import *
print("Setup Complete")


# You will work with data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course) to predict home prices in Iowa using 79 explanatory variables describing (almost) every aspect of the homes.  
# 
# ![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)
# 
# Run the next code cell without changes to load the training and validation features in `X_train` and `X_valid`, along with the prediction targets in `y_train` and `y_valid`.  The test features are loaded in `X_test`.  (_If you need to review **features** and **prediction targets**, please check out [this short tutorial](https://www.kaggle.com/dansbecker/your-first-machine-learning-model).  To read about model **validation**, look [here](https://www.kaggle.com/dansbecker/model-validation).  Alternatively, if you'd prefer to look through a full course to review all of these topics, start [here](https://www.kaggle.com/learn/machine-learning).)_

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


# Use the next cell to print the first several rows of the data. It's a nice way to get an overview of the data you will use in your price prediction model.

# In[4]:


X_train.head()


# # Step 1: Evaluate several models
# 
# The next code cell defines five different random forest models.  Run this code cell without changes.  (_To review **random forests**, look [here](https://www.kaggle.com/dansbecker/random-forests)._)

# In[5]:


from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]


# To select the best model out of the five, we define a function `score_model()` below.  This function returns the mean absolute error (MAE) from the validation set.  Recall that the best model will obtain the lowest MAE.  (_To review **mean absolute error**, look [here](https://www.kaggle.com/dansbecker/model-validation).)_
# 
# Run the code cell without changes.

# In[6]:


from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))


# Use the above results to fill in the line below.  Which model is the best model?  Your answer should be one of `model_1`, `model_2`, `model_3`, `model_4`, or `model_5`.

# In[9]:


# Fill in the best model
best_model = model_3

# Check your answer
step_1.check()


# In[10]:


# Lines below will give you a hint or solution code
#step_1.hint()
#step_1.solution()


# # Step 2: Generate test predictions
# 
# Great. You know how to evaluate what makes an accurate model. Now it's time to go through the modeling process and make predictions. In the line below, create a Random Forest model with the variable name `my_model`.

# In[11]:


# Define a model
my_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=42) # Your code here

# Check your answer
step_2.check()


# In[12]:


# Lines below will give you a hint or solution code
#step_2.hint()
#step_2.solution()


# Run the next code cell without changes.  The code fits the model to the training and validation data, and then generates test predictions that are saved to a CSV file.  These test predictions can be submitted directly to the competition!

# In[13]:


# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)


# # Step 3: Submit your results
# 
# Once you have successfully completed Step 2, you're ready to submit your results to the leaderboard!
# - Begin by clicking on the blue **COMMIT** button in the top right corner.  This will generate a pop-up window.  
# - After your code has finished running, click on the blue **Open Version** button in the top right of the pop-up window.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# - Click on the **Output** tab on the left of the screen.  Then, click on the **Submit to Competition** button to submit your results to the leaderboard.
# - If you want to keep working to improve your performance, select the blue **Edit** button in the top right of the screen. Then you can change your model and repeat the process.

# # Keep going
# 
# You've made your first model. But how can you quickly make it better?
# 
# Learn how to improve your competition results by incorporating columns with **[missing values](https://www.kaggle.com/alexisbcook/missing-values)**.

# ---
# **[Intermediate Machine Learning Micro-Course Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# 
