#so what is data leakage?
'''this is when your training data contains information about the target,but similar data will not be available
when the model is used for production.There are two main types of data leakage:
a)target leakage
b)train-test contamination
1)target leakage occurs when your predictors will not be available at the time to make predictions.It is important 
to think of target leakage as the timing or chronological order not merely whether a feature helps make good decisions.
look at how the feature that you are giving is directly related to what you want to achieve(target) and you will realise 
that if a feature (is functionally dependent() on target and the inverse
to prevent data leakage  any variable updated after the target value is realized should be excluded.
2)train_test contamination-