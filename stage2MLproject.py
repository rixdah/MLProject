import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn import preprocessing

def linRegr(X, y, lossFunc):
    # Split data into training, validation and test data
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.5, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.25, random_state=42)
    print("X_train length: "+str(len(X_train)))
    print("X_val length: "+str(len(X_val)))
    print("X_test length: "+str(len(X_test)))
    lin_regr = LinearRegression()
    lin_regr.fit(X_train, y_train)
    y_pred_train = lin_regr.predict(X_train)
    y_pred_val = lin_regr.predict(X_val)
    y_pred_test = lin_regr.predict(X_test)

    if (lossFunc == 'MSE'):
        MSEtrain_error = mean_squared_error(y_train, y_pred_train)
        MSEval_error = mean_squared_error(y_val, y_pred_val)
        MSEtest_error = mean_squared_error(y_test, y_pred_test)
        return MSEtrain_error, MSEval_error, MSEtest_error

    return 'Wrong loss function given'

def logRegr(X, y):
    # Split data into training, validation and test data
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.5, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.25, random_state=42)
    log_regr = LogisticRegression()
    log_regr.fit(X_train, y_train)
    y_pred_train = log_regr.predict(X_train)
    y_pred_val = log_regr.predict(X_val)
    y_pred_test = log_regr.predict(X_test)
    LOGtrain_error = log_loss(y_train, y_pred_train)
    LOGval_error = log_loss(y_val, y_pred_val)
    LOGtest_error = log_loss(y_test, y_pred_test)
    print("Accuracy of logistic regression with validation set: ", accuracy_score(y_val, y_pred_val))
    print("Accuracy of logistic regression with test set: ", accuracy_score(y_test, y_pred_test))

    return LOGtrain_error, LOGval_error, LOGtest_error

df = pd.read_csv('student-mat.csv')

# Get the final grade (G3) as the label
y = df['G3']

# The mean of the final grades
G3mean = y.mean()
y = pd.DataFrame(y)

# Get the chosen features
X = df[['address', 'studytime', 'failures', 'schoolsup', 'freetime', 'Medu', 'absences', 'Dalc', 'Walc']]

# Change binary values to 0 and 1
le = preprocessing.LabelEncoder()
binary_features = X.select_dtypes("object").columns
for feature in binary_features: 
    X[feature] = le.fit_transform(X[feature])
print(X.head())

# Get mean squared errors for training, validation and test data
MSEtrain, MSEval, MSEtest = linRegr(X, y, 'MSE')
print('Mean squared errors without G1 and G2:')
print('Linear regression training error: ' + str(MSEtrain))
print('Linear regression validation error: ' + str(MSEval))
print('Linear regression test error: ' + str(MSEtest))

# Add G1 and G2 to chosen features
X_with_grades = df[['address', 'studytime', 'failures', 'schoolsup', 'freetime', 'Medu', 'absences', 'Dalc', 'Walc', 'G1', 'G2']]

# Change binary values to 0 and 1
le = preprocessing.LabelEncoder()
binary_features = X_with_grades.select_dtypes("object").columns
for feature in binary_features: 
    X_with_grades[feature] = le.fit_transform(X_with_grades[feature])
print(X_with_grades.head())

# Do linear regression with G1 and G2 as features
MSEtrain, MSEval, MSEtest = linRegr(X_with_grades, y, 'MSE')
print('Mean squared errors with G1 and G2:')
print('Linear regression with grades training error: ' + str(MSEtrain))
print('Linear regression with grades validation error: ' + str(MSEval))
print('Linear regression with grades test error: ' + str(MSEtest))

# Convert label (final grade G3) to 1, if it is higher or equal to the average final grade and to 0, if it is lower
# This is used for logistic regression

y['G3'] = np.where(y['G3'] >= G3mean, 1, 0)

# Do logistic regression without G1 and G2 as features
LOGtrain, LOGval, LOGtest = logRegr(X, y)
print('Logistic loss without G1 and G2:')
print('Logistic regression training error: ' + str(LOGtrain))
print('Logistic regression validation error: ' + str(LOGval))
print('Logistic regression test error: ' + str(LOGtest))

# Do logistic regression with G1 and G2 as features
LOGtrain, LOGval, LOGtest = logRegr(X_with_grades, y)
print('Logistic loss with G1 and G2:')
print('Logistic regression with grades training error: ' + str(LOGtrain))
print('Logistic regression with grades validation error: ' + str(LOGval))
print('Logistic regression with grades test error: ' + str(LOGtest))