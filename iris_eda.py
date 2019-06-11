
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
palette is the colors being blue and red since the target variable is encoded with two colours as they are two '''


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


# c=color thus ensuring that our data points are coloured by the target
# figsize = size of the figure
# s=shape
# marker size
