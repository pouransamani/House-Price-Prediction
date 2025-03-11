# House-Price-Prediction
## Regression method using Lasso, RandomFarest, XGBoost, and a detailed EDA
## Table of Content

- [Executive Summary](#Executive-Summary)
- [Introduction](#Introduction)
- [Loading libraries and reading the data](#Loading-libraries-and-reading-the-data)
- [preproccessing and Data Cleaning](#preproccessing-and-Data-Cleaning)
- [Exploratory Data Analysis(EDA)](#preproccessing-and-Exploratory-Data-Analysis-(EDA))
- [Feature Engineering](#Feature-Engineering)
- [Model Development](#Model-Development)
- [Feature Importance](#Feature-Importance)
- [Voting Classifier](#Voting-Classifier):
- [Model Tuned performance](#Model-Tuned-Performance)
- [Conclusions](#Conclusions)
- [Recommendations](#Recommendations)



  ## Executive Summary
  - This project started by focusing on getting a good understanding of the dataset. The EDA is detailed and many visualizations are included and followed by modeling. 
    * The XGBoost model performs best with an average CV R2 Score of 0.8869. 
    * Random Farest performs well with an average CV R2 Score of 0.8743. 
    * Lasso regression stands on third position with an avarage CV R2 score of 0.8701.
    * As the algorithms are very different, Votingregressor was used for averaging predictions to improve the predictions. 
 
  ## Introduction
  - A home buyer describe his dream house. He might focus on the number of bedrooms or the beautiful yard. However, as it will discover in this project, many factors can influence a home's price beyond just its visible features. With a dataset containing 79 variables that describe nearly every aspect of residential properties in Ames, Iowa, the challenge will be to predict the final sale price of each home.
  - This project will give hands-on experience in:
  - Creative Feature Engineering: Identifying and creating new features that could improve model’s performance.
  - Advanced Regression Techniques: Implementing and tuning models such as Random Forests and Gradient Boosting to make accurate predictions.

## Loading libraries and reading the data
- This section loads the train and test datasets using pandas.  
- The train_test column created to differentiate between training and test data.
- The test data does not include the actual SalePrice column in the test dataset, it is set to NaN because this is what we need to predict. Adding this column ensures consistency in the structure of both datasets, making them easier to combine.  
- Both datasets concatenated into a single dataframe for `consistent preprocessing`.
**1- Import libraries and dataset**
- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt
- import seaborn as sns
- import datetime
- from sklearn.preprocessing import StandardScaler
- from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
- import sklearn.linear_model as linear_model
- from sklearn.metrics import mean_squared_error, r2_score
- from sklearn.preprocessing import LabelEncoder
- from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
- from sklearn.inspection import permutation_importance
- from scipy.stats import  johnsonsu , zscore, skew, boxcox_normmax  # for some statistics
- from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso
- from sklearn.svm import SVR
- from sklearn.pipeline import make_pipeline
- from mlxtend.regressor import StackingCVRegressor
- from xgboost import XGBRegressor
- import xgboost as xgb
- from lightgbm import LGBMRegressor
- import warnings
- warnings.filterwarnings("ignore")

## Loading Dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#### Assign identifiers
train['train_test'] = 1  
test['train_test'] = 0  
#### Assign NaN to test SalePrice
test['SalePrice'] = np.nan   # Assign NaN to missing target variable in the test set
 - There are 1460 instances of train data and 1459 of test data. Total number of attributes equals 81, from which 36 is quantitative, 43 categorical + Id and SalePrice.
- Quantitative-numerical features are as follow: 
  - ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
 'MoSold', 'YrSold']
 - Qualitative-categorical features are as follow:
   - ['MSZoning',  'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
 
## preproccessing and Data Cleaning
 - Handling Duplicat
   - `There is no duplicate data`
 - Handling missing values
    - **Insights from missing data**
     - According to the bar chart: 35 features have missing values, 6 features are 50% and above, 23 features are below 5% and 6 features are between (5-20)%.
  ![image alt](https://github.com/pouransamani/House-Price-Prediction/blob/52536cb308286c785b0d35f79109cc26650deb7f/missing%20value%20bar.png)
   
     - Missing data was computed according to median and mode methods.
     - To fix the 35 predictors that contains missing values. When I go through each feature having NAs there are multiple variables that relate to Pool, Garage, and Basement, so I deal with them as a group referring to meta data which shows definiton of variables and make it clear that NA in that feature doesn't mean missed value, it means that house dosn't have Pool, Garage, or Basement, so it will be replaced with NONE.
     - listing features having NAs values
     - Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PoolQC and Fence.

## preproccessing and Exploratory Data Analysis (EDA)
Data Cleaning:
- Remove duplicate entries.
- Handle missing values.
- Correct inconsistencies in the data.
- Detect and handle outliers.
- Generate summary statistics for the dataset.
- Visualize the distribution of each feature.
- Explore relationships between features using scatter plots, correlation matrices(Heatmap), etc.
- Identify any patterns or trends in the data.

## Feature Engineering:
- Normalize target.
- Using P-Values, chi-square test for identifying statistically significant features. 
- Encode categorical variables.

## Model Development:
- Use the cleaned and preprocessed dataset to build a classification model using DT, LR, KNN, SVM, RF algorithms.
- Split the dataset into training and testing sets.
- Train the model with package of algorithms and evaluate its performance using accuracy measure.
- Explore how model performance changes with different features or transformations using Cross Validation.
- Tuning the model by GridSearchCV
- Summarize findings and assess the models accuracies in classification personal loan receiving.
## Feature Importance:
- Use feature importance technique to identify the most influence varaibles and enhancing models performance.
-For every single algorithm;
   - Tree-based models (Random Forest, and Decision Tree) → Importance based on feature splits.
   - Permutation Importance → Measures how model accuracy changes when a feature is shuffled.
   - Coefficients in Linear Models (Logistic Regression, Linear SVM) → Shows feature weight in prediction.

## Voting Classifier:
 - Combining predictions from different applied machine learning models, provide an average prediction result based on the prediction of all the submodels. 

|Model |Baseline|Cross Validation|Tuned Performance(GridSearchCV)|Voting Classifier(Ensemble Methode)|
|------|--------|----------------|------------------------------ |-----------------------------------|
|Decision Tree| 96.73%|97.20% |97.71% ||
|Logistic Regression| 95.2%|95.62% |95.68% ||
|K Nearest Neighbor| 96.93%|96.82%| 97.11%||
|rbf-SVM| 97.86%|97.57% |97.97% ||
|linear-SVM|95.26%| 95.65%  | --||
|Random Forest| 98.20%|98.14%   |98.25% ||
|Ensemble (Voting Classifier) ||98.17%||                                       97.93% |

## Conclusion:
- The Random Forest model identifies Education, Income, Mortgage, CCAvg, and Family as the most influential factors in predicting personal loan applications.
- Understanding the impact of these features helps banks refine the model and gain a deeper understanding of applicants' financial behavior.

## Recommondation:
- The key factors influencing the prediction of personal loan applications are Education, Income, Mortgage, CCAvg, and Family. These features provide valuable insights that help banks make informed decisions about loan approvals and shape their marketing strategies. By considering these factors, banks can better assess applicants' eligibility and target potential customers more effectively. 

 🙂 💻
     
