#pipelining is a good way to keep data preprocessing and modelling code organize
#when using pipelines you achieve cvleaner code,fewer bugs,easier to productionaze,more options for model validation
#example of the same 
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
#part one is to define the preprocessing process
'''Similar to how a pipeline bundles together preprocessing and modeling steps,
 we use the ColumnTransformer class to bundle together different preprocessing steps
 1)imputing the missing values in the numerical data
 2)imputes missing values and applies a one-hot encoding to categorical data.
 when using the simple imputer this is what different parameters stand for:
If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
If “most_frequent”, then replace missing using the most frequent value along each column.
 Can be used with strings or numeric data.
If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
 '''
 from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')
#this replaces the zero or null data points with zeros
#and then for the no using this replaces them with Nans

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
 2)
#now lets define the model 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=0)
#now the last step is to create and evaluate the pipeline(pipeline is a class)
'''With the pipeline, we preprocess the training data and fit the model in a single line of code.
 (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps.
  This becomes especially messy if we have to deal with both numerical and categorical variables!)
With the pipeline, we supply the unprocessed features in X_valid to the predict() command, 
and the pipeline automatically preprocesses the features before generating predictions. 
(However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)
'''
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
'''when you want to call several models together one by one and compare their individual_results you just 
use a simple pipeline and this simplifies the whole process
'''