
# coding: utf-8

# In[18]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[19]:


#importing necessary dependencies plus dataset
from sklearn import datasets
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris=datasets.load_iris()
#when using data that is binary to get insights this is how it is done
'''plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
where the target_variable is the party and the x is the feature you want to put under the microscope
palette is the colors being blue and red since the target variable is encoded with two colours as they are two
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
#looking at the MAE(basically the mean value from the predicted value) and we know that validation is just
comparing the actual value with the test datapoints
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices) '''


# In[20]:


type(iris)


# a bunch is similar to dictionary in that it contains key value pairs

# In[21]:


print(iris.keys())


# In[22]:


#looking at the target and the data types themselves
type(iris.data),type(iris.target)


# so both are arrays

# In[23]:


iris.data.shape


# lookin at this we can detrermine that we have 150 samples and 4 features

# In[24]:


#look at the target names
iris.target_names


# In[25]:


#assigning the data and features to variables
x = iris.data
y = iris.target


# In[26]:


#building a data frame 
df = pd.DataFrame(x,columns=iris.feature_names)


# In[27]:


df.head()


# > using visual EDA

# In[29]:


_=pd.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='D')
'''all machine learning models are implemented as python classes
models require that the training data for fitting be either a data frame or numpy array
you can always use for example
prediction=knn.predict('') 
print('Prediction {}'.format(prediction))
the accuracy as a metric is the no of correct predictions over the total predictions
this is the work of the test set after you split the data 
after spliiting the data adnd training it on the train data then predicting on the x_test now
look at the pred to see the format if it is the one that you expected
print('Test set predictions:\n {}'.format(y_pred))
now on the accuracy we use the score method of the model and pass in the test_split
knn.score(X_test,y_test)
larger values for the k_neighbors means a smoother and less complex model while bigger 
values increase the complexity
but do not use a very large k as that means that we will be underfitting  '''


# c=color thus ensuring that our data points are coloured by the target
# figsize = size of the figure
# s=shape
# marker size
