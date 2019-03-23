#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Load Boston dataset from skit-learn
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from sklearn.datasets import load_boston
boston = load_boston()
# print(type(boston))
# print(boston)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Exploratory Data Analysis (EDA)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Print the properties of Boston Dataset
print(dir(boston))
# Output: ['DESCR', 'data', 'feature_names', 'filename', 'target']

# Print the data type and data of each properties of Boston data
for directory in boston:
    print("Directory Name: ", directory)
    # print(type(boston[directory]))
    print(boston[directory])

# Convert Numpy array to Pandas Data Frame
dataset = pd.DataFrame(boston.data)
# Add Column Names in the Dataset
dataset.columns = boston.feature_names
print(dataset.head(10))

# Target value MEDV / Price is missing from the dataset.
# Create a new column MEDV in the dataset and add target values.
dataset['MEDV'] = boston.target
print(dataset.head(10))

# Shape of the dataset
print("Dataset Shape is: ", dataset.shape)

# Compute and display summary statistics
print("Summary Statistics")
print(dataset.describe())
# --------------------------------------------------------------
# Create a directory for saving diagrams
if not os.path.exists('plots'):
    os.mkdir('plots')
# --------------------------------------------------------------
# Visualize Histogram Plot for 'target' feature that is MEDV
# Create and save Histogram
fig = plt.figure(figsize=(10,7.5))
plt.grid(axis='y', alpha=0.5)
ax = sns.distplot(dataset['MEDV'], bins=30, hist_kws=dict(edgecolor="w", linewidth=2))
ax.set_title('Histogram', fontsize=20)
ax.set_xlabel('MEDV or Price', fontsize=20)
ax.set_ylabel('Frequency', fontsize=20)
plt.savefig("plots/" + "histMEDV.png", dpi=70)
plt.close(fig)
#Findings: Values of MEDV are distributed normally with few outliers.
# -----------------------------------------------------------------
# Visualize Heatmap of the Dataset
# Create and save Heatmap
fig = plt.figure(figsize=(10,7.5))
ax = sns.heatmap(dataset.corr(method = "pearson"), annot=True, cmap='coolwarm', linewidth=0.5)
ax.set_title('Pearson Heat Map', fontsize=20)
plt.savefig("plots/" "PearsonHeatMap.png", dpi=70)
plt.close(fig)
# plt.show()
# Findings: RM and LSTAT have high correlation with MEDV
# -----------------------------------------------------------------
# Visualize Scatter Diagram pairing with MEDV feature
column = len(dataset.columns)

for x in range(column-1):
    fig = plt.figure(figsize=(10,7.5))
    ax = sns.scatterplot(dataset.iloc[:,x], dataset['MEDV'], edgecolors='w', alpha=0.7)
    ax.set_title('Scatter Plot', fontsize=20)
    ax.set_xlabel(str(dataset.columns[x]), fontsize=20)
    ax.set_ylabel('MEDV or Price', fontsize=20)
    plt.savefig("plots/" + "scatter" + "-" + str(dataset.columns[x]) + "-"+ "MEDV.png", dpi=80)
    plt.close(fig)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Apply Machine Learning Algorithms for Predicting MEDV / Prices
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Create Datasets for Target and Predictors
# Target value Y and Predictor values X. Thus, Y = Price, X = RM & LSTAT features
# Create dataset with RM and LSTAT
X = pd.DataFrame(np.c_[dataset['LSTAT'], dataset['RM']], columns = ['LSTAT','RM'])
Y = dataset['MEDV'] # Create dataset with MEDV

# Split the dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
print("X Train: ", X_train.shape)
print("Y Train: ", Y_train.shape)
print("X Test: ", X_test.shape)
print("Y Test: ", Y_test.shape)
# --------------------------------------------------------------
# Linear Rigression
# Split the dataset into train and test
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test) # Prediction

# Create Dataset with Testing values and Predicted Prices
print("Linear Regresson Model")
model_lr = pd.DataFrame(X_test)
model_lr['MEDV'] = Y_test
model_lr['Predicted MEDV'] = Y_pred
print(model_lr.head(10))

# Measure Performance of the Model
# Get Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, Y_pred)
# Get Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, Y_pred)

err = "MSE: " + str(round(mse, 2)) + "," + " MAE: " + str(round(mae, 2))
print("Linear Regression Model Performance:-- ", err)

fig = plt.figure(figsize=(12,9))
ax = sns.regplot(Y_test, Y_pred, marker = 'o', color = 'green')
ax.set_title('Linear Regrassion', fontsize=20)
ax.set_xlabel('MEDV or Price', fontsize=20)
ax.set_ylabel('Predicted Prices', fontsize=20)
# Save the Linear Regrassion Plot along with Error value
plt.text(35.0, 0.0, err, fontsize=20, bbox=dict(facecolor='green', alpha=0.5))
plt.savefig("plots/" + "LinearRegression.png", dpi=70)
plt.close(fig)
# --------------------------------------------------------------------
# KNN algorithm Training and Predictions
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=1)
knr.fit(X_train, Y_train)

Y_pred = knr.predict(X_test) # Prediction

# Create Dataset with Testing values and Predicted Prices
print("KNN Regresson Model")
model_knn = pd.DataFrame(X_test)
model_knn['MEDV'] = Y_test
model_knn['Predicted MEDV'] = Y_pred
print(model_knn.head(10))

# Measure Performance of the Model
# Get Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, Y_pred)
# Get Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, Y_pred)

err = "MSE: " + str(round(mse, 2)) + "," + " MAE: " + str(round(mae, 2))
print("KNN Regresson Model Performance:-- ", err)

# Create Regression Plot for Test and Prediction values
fig = plt.figure(figsize=(12,9))
ax = sns.regplot(Y_test, Y_pred, marker = 'o', color = 'blue')
ax.set_title('KNN Regrassion', fontsize=20)
ax.set_xlabel('MEDV or Price', fontsize=20)
ax.set_ylabel('Predicted Prices', fontsize=20)
# Save the KNN Regrassion Plot along with Error value
plt.text(35.0, 10.0, err, fontsize=20, bbox=dict(facecolor='blue', alpha=0.5))
plt.savefig("plots/" + "KNNRegression.png", dpi=70)
plt.close(fig)
# -------------------------------------------------------------------
# Gradient Boosting Tree Regression
# Split the dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)

Y_pred = gbr.predict(X_test) # Predictions

# Create Dataset with Testing values and Predicted Prices
print("Gradient Boost Regresson Model")
model_gbr = pd.DataFrame(X_test)
model_gbr['MEDV'] = Y_test
model_gbr['Predicted MEDV'] = Y_pred
print(model_gbr.head(10))

# Measure Performance of the Model
# Get Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, Y_pred)
# Get Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, Y_pred)

err = "MSE: " + str(round(mse, 2)) + "," + " MAE: " + str(round(mae, 2))
print("Gradient Boosting Regrassion Model Performance:-- ", err)

fig = plt.figure(figsize=(12,9))
ax = sns.regplot(Y_test, Y_pred, marker = 'o', color = 'r')
ax.set_title('Gradient Boosting Regrassion', fontsize=20)
ax.set_xlabel('MEDV or Price', fontsize=20)
ax.set_ylabel('Predicted Prices', fontsize=20)
# Save the KNN Regrassion Plot along with Error value
plt.text(35.0, 10.0, err, fontsize=20, bbox=dict(facecolor='r', alpha=0.5))
plt.savefig("plots/" + "GradientBoostingRegression.png", dpi=70)
plt.close(fig)
# ---------------------------------------------------------------
# Model Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# Cross validate model with Kfold stratified cross val
# fold = KFold(n_splits=10, shuffle=True, random_state = 2)
# print(fold)

# Modeling step Test differents algorithms
# random_state = 2
# regressors = []
# regressors.append(LinearRegression())
# regressors.append(KNeighborsRegressor())
# regressors.append(GradientBoostingRegressor())

print("Gradient Boost Regression Cross Validation with KFold:")
kfolds = [2, 3, 4, 5]
for kfold in kfolds :
    cv_results = cross_val_score(gbr, X, Y, cv = KFold(kfold, random_state = 0))
        # print(cv_results)
    # print(cv_results.mean())
    print("Kfold = ", kfold, ", MAE ", round(cv_results.mean(), 2))

# regressors = [lr, knr, gbr]
# kfolds = [2, 3, 4, 5]
# for kfold in kfolds :
#     print ("Kfold ", kfold)
#     for regressor in regressors :
#         cv_results = (cross_val_score(regressor, X, Y, cv = KFold(kfold, random_state = 0)))
#         if regressor == lr :
#             print("Linear Regression, MAE: ", cv_results.mean())
#         elif regressor == knr:
#             print("KNN Regression, MAE: ", cv_results.mean())
#         else:
#             print("Gradient Boost Regression, , MAE: ", cv_results.mean())
