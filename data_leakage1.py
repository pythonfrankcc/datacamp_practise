#detect and remove data leakages 
import pandas as pd

# Read the data
data = pd.read_csv('../input/aer-credit-card-data/AER_credit_card_data.csv', 
                   true_values = ['yes'], false_values = ['no'])
#the true value converts available yes strings to True and the false converts the No strings in data to False
# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
X.head()
'''since the dataset is small we will use cross-validation to determine the model quality and use a pipe as a best
practise'''
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())
'''Here is all the relevant data on the columns,ie, columns used
card: 1 if credit card application accepted, 0 if not
reports: Number of major derogatory reports
age: Age n years plus twelfths of a year
income: Yearly income (divided by 10,000)
share: Ratio of monthly credit card expenditure to yearly income
expenditure: Average monthly credit card expenditure
owner: 1 if owns home, 0 if rents
selfempl: 1 if self-employed, 0 if not
dependents: 1 + number of dependents
months: Months living at current address
majorcards: Number of major credit cards held
active: Number of active credit accounts 

We get to see that the model has an acccuracy of 97% which is very alarming meaning that the data that we have may
have some leakage and our model may be interpreting some data with a certain level of correlation that may be 
detrimental during production
**look at the data that my be ambiguous such as the expediture column does expenditure mean expenditure on the 
card given or the cards that one had before receiving this card
'''
#let us do a  basic data comparison the expenditure and the target,card.
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
#This is what we get'
'''Fraction of those who did not receive a card and had no expenditures: 1.00
Fraction of those who received a card and had no expenditures: 0.02'''
#from this it can be seen that the model may have overlooked the fact that those who also have cards may have 0 expenditure
'''On getting a leaky predictor also look at the features that have a relationship with the leaky feature
for example in this we know that share has a direct relationship with the leaked feauture expenditure.
The ambiguous feaures such s  majorcards and active can also be removed with this lit is better to be safe than 
sorry so you can keep removing the feautures that you suspect as you check how they affect the accuracy of the model
(there is no just right and wrong when it comes to data leakage)
'''
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())
#with this you note the models accuracy drops but atleast does well on the test data
