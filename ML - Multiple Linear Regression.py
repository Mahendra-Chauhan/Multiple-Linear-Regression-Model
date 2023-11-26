### MULTIPLE LINEAR REGRESSION ###

link = "D:\\TOP MENTOR\\MachineLearning-master\\MachineLearning-master\\3_Startups.csv"
import pandas as pd
df = pd.read_csv(link)
print(df)

# EDA MEANS PLOT THE GRAPH
# EXPLORATORY DATA ANALYSIS.
import matplotlib.pyplot as plt
plt.scatter(df["R&D Spend"],df["Profit"])
plt.show()  # Strong correlation is here.

plt.scatter(df["Administration"],df["Profit"])
plt.show() # some correlation but not strong as above

plt.scatter(df["Marketing Spend"],df["Profit"])
plt.show() # Some correlation higher than the administration but not as strong as R&D spend.

# Getting X and Y value to handle the problems.
X = df.iloc[:,:4].values
y = df.iloc[:,4].values

#Handling categorical value
print("Before handling: \n",X[:,3])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #OneHotEncode is used to get the column transform on the left.
lc = LabelEncoder()
X[:,3] = lc.fit_transform(X[:,3])
print("After encoding: \n",X[:,3])
from sklearn.compose import ColumnTransformer
trans = ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[3])],remainder="passthrough")
X = trans.fit_transform(X)
X = X[:,3:]  #delete column 1
print("After deleting one column",X)

## break it into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.25)

### RUN THE MODEL
## Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#
print("Coefficient = ",regressor.coef_)
print("Intercept = ",regressor.intercept_)
'''
Coefficient =  [-3.46665014e+03 -3.26326437e+03  8.57992247e-01 -7.96539318e-03
  1.74803461e-02]
Intercept =  47507.358988965905
y = -3466.67 x1 - 3263.2 x2 + 0.858 x3 - 0.00797 x4 + 0.0175 x5 + 47507.4

'''
## testing the model with test/validation data
y_pred =regressor.predict(X_test)
result = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(result)

# Evaluation
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
print("VALIDATION / VARIANCE: Mean Absolute Error = ",mae)
mse = metrics.mean_squared_error(y_test, y_pred)
print("VALIDATION: Mean Squared Error = ",mse)
rmse = mse**0.5
print("VALIDATION: Root Mean Squared Error = ",rmse)
r2 = metrics.r2_score(y_test, y_pred)
print("VALIDATION: R Squared value = ",r2)


'''
r2 =0 means regression line = avegrae line, no diffrence at all
r2 =1 mens there is no error perfect regression line. it is the best fit line where all the point
are lies on the regression line.

We want mae, mse, rmse, r2 these all closer to zero. means no erro, erros as closes to be zero.
R2 irrestive of any dataset will be between 0 to 1. other than that will changes but wnat to be nearer to zero.
so the error can be minimized,

mae, mse, rmse for same data set
r2 irrestive of any data set

VARIANCE :- Checking on the validation error, test error. high error means high varinace
BAIS :- Checking the traning error - high error means high bias.
Bias is low means training is low
Varinace is high, test data is high, overfitting, overconfidence, memorizing the things
Overfitting means your model stop thinking, and whaever mughup he will tell only that.
Underfitting means not giving enough training to the data. He has low knowledge.
so in underfitting, you dont know how to run the model, and does not know any parameter.
 
Suppose we have 10 values for x and 1 for Y, it will be 11th dimensional.
We have 3 values for x and 1 for Y, it will be 4 dimensional.

1 feature =  means 1 value for x
2 features=  means 2 values for x like x1 and X2.


'''