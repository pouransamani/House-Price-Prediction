# House-Price-Prediction
## Regression method using Lasso, RandomFarest, XGBoost, and a detailed EDA
## Table of Content

- [Executive Summary](#Executive-Summary)
- [Introduction](#Introduction)
- [Loading libraries and reading the data](#Loading-libraries-and-reading-the-data)
- [preproccessing and Data Cleaning](#preproccessing-and-Data-Cleaning)
- [Exploring Data Analysis (EDA) and Data Wrangling](#Exploring-Data-Analysis-(EDA)-and-Data-Wranglin)
- [Feature Engineering](#Feature-Engineering)
- [Modeling](#Modeling)
- [Feature Importance](#Feature-Importance)
- [Voting Regressors](#Voting-Regressors):
- [Model performance](#Model-Performance)
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
 - Assign identifiers
train['train_test'] = 1  
test['train_test'] = 0  
 - Assign NaN to test SalePrice
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



## Exploring Data Analysis (EDA) and Data Wrangling
- Visualize the distribution of each feature.
   - Histplot
     - **Insights for Histogram charts**
       - Features with no significant or atypical statistical distribution; They could be categorical features like `MSSubClass`(I converted this feature to catfeature in **Feature Engineering**), or have a high concentration of zeros, or have very limited variation across values like Pool, Garage and Fireplace groupe features. 
         - `MSSubClass`, `BsmtFinSF2`, `LowQualFinSF`, `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, `BedroomAbvGr`, `KitchenAbvGr`, `Fireplace`, `GarageCars`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `PoolArea`, `MiscVal`, `MoSold`,`YrSold`
     - Features with normal distribution;
       - For `LotFrontage` might be influenced by many factors such as location, zoning laws, or the size of neighboring properties, and the combined effect of all these factors leads to a normal distribution.
       -  `TotalBsmtSF`, `2ndFlrSF`: Aggregated values such as the total square footage of a house or the sum of individual room areas (like TotalBsmtSF), might also exhibit normality.
     - Features with skewness to right; it shows that the distribution has a longer tail on the right, and has logical relationship with price: Larger values for features like GrLivArea and TotRmAbvGrd often correspond to larger homes, which typically command higher prices.  
       - `OveralQual`,  `LotArea`, `MasVnrarea`, `BsmtFinSF1`, `BsmtUnfSF`, `1stFlrSF`, `GrLivArea`, `TotRmabvGrd`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `SalePrice`  
     - Features with skewness to left; it shows that the distribution has a longer tail on the left side. For features like construction and renovation years, homes tend to be relatively newer in the datasets. This results in more data points being clustered around recent years (higher values) and fewer older properties contribute to the tail on the left. Since in the real state market , they talk about age of the house, I figured out to do some **Feature Engineering** and Convert year to age.  
      - `YearBlt` , `YearRemodAdd`, `GarageYrBlt`
 
- Correlation matrices(Heatmap).
  ![image alt](https://github.com/pouransamani/House-Price-Prediction/blob/e2488c9670c50007abffcd8ab47923e96c0b005b/correlation.png)
  
- Scatter-Regression plot.
  
     - **Note resulted form Scatter-regression plots**
         - Features show `no relation` with Price;
           - MSSubClass, LowQualFinSF, BmtFinSF2, BsmtHalfBath, 3SsnPorch, MiscVal, MoSold, YrSold, ScreenPorch,
         - Features with `positive relation with price, high slop regression line`considered as highly important features;
           - GrLivArea (Above grade living area), OveralQual, totalBsmtSF(Total square feet of basement area),TotRmsAbvGrd, 1stFlrSF(First Floor square feet), BsmtFinishSF1, LotFrontAge,LotArea, MasVnrArea
           - the recent year the higher price, `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`

         - Features with `positive relation, gentel slop regression line`considered as importance features;
            -  BsmtFullBath, FullBath(Full bathrooms above grade), HalfBath, BedroomAbvGr, FirePlace, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, 2ndFlrSF, BsmtUnfSF, PoolArea

        - If year related features be converted to Age, so these Features will present `negative relation, gentel slop regression line`considered as importance features; 
           - Age_housBlt, Age_RemodAdd, Age_GargeBlt , KitchenAbvGr, EnclosedPorch 

        - Regarding outliers, except two data points, extreme values are not seen. If there is a candidate to take out as an outlier later on, it seems to be the expensive house. It is seen two data points on Lotfrontage, and GrLivArea.

        - It also becomes clear the `multicollinearity` is an issue. For example:
            - GarageCars and GarageArea show the same correlations with SalePrice and high correlation between them (0.89).
            - Age_houseBlt, Age_RemodAdd, Age_GargeBlt show the same issue.
          ![image alt](https://github.com/pouransamani/House-Price-Prediction/blob/5ad3b7abf6b2392863fda73e8cd87b64b140d46e/reg-Scatter%20plot.png)
- Scatter plot to show extrem house price
In order to better performance of model, I did visualize price on Scatter plot through two features to find max values.
![image alt](https://github.com/pouransamani/House-Price-Prediction/blob/0a1a80323cc8055b2ff47f1b52d2036bea4dd680/extreme%20price.png)

I droped these two extreme values.
![image alt](https://github.com/pouransamani/House-Price-Prediction/blob/b8a3075f675021847ceac0435ba351d72acc0612/drop%20extreme%20price.png)
- Identify any patterns or trends in the data.

## Feature Engineering:
  - refering to histogram plot, features with no relationship that could be convert to categorical feature, as weel as in order to make the dataset more meaningful and well-structured for EDA, I replaced the numerical encoded values of features  MSSubClass and MoSold to their Real State terms.
  - Normalize target.
  - Encode categorical variables into ordinal and dummy variables.

## Modeling
Use the cleaned and preprocessed dataset to build the Regression model using LinearRegression, Ridge, Lasso, Random Farest and XGBoost algorithms.
Explore how model performance changes with different features/or transformed using Cross Validation.
 - Base Model_Linear Regression
 - Cross validation with advanced models
 - 
 ## Model performance
 - Evaluate models performance using RMSE and R2 Score measures.
 - Summarize findings and assess the models performanc in predicting house price.
   
![image](https://github.com/user-attachments/assets/fbd99d88-9e08-4276-babc-6800976da2b4)

## Feature Importance:
- Use feature importance technique to identify the most influence varaibles and enhancing models performance.
- Feature selection helps improve model performance by keeping only the most relevant features. Here are some common techniques I used:
  - a. Correlation Analysis
Remove features highly correlated with each other. Keep features that have a good correlation with the target variable.
![image alt](https://github.com/pouransamani/House-Price-Prediction/blob/b90e18279324183df4c46060ccd49731b5c25ddf/Feature%20importance-Corr.png)
  - b. Feature importance  using Algorithms; SelectKBest--f_regression,  Lasso Embedded method, Random Farest, XGBoost, Top Features Selected by Permutation Importance (SVR), and LightGBM.
![image alt](https://github.com/pouransamani/House-Price-Prediction/blob/6e082896c75dbd8cd2689bbbf66e309be2c114e3/feature_importances_XGB.png)

## Voting Regressors-Final Submission:
 - Combining predictions from different applied machine learning models, provide an average prediction result based on the prediction of all the submodels.
   
![image](https://github.com/user-attachments/assets/dc4517d1-f411-4278-8f59-ea19e37fc199)


## Conclusion:
- Model Performane:
  - Lasso Regression is the best choice for predictive accuracy (low RMSE) while also performing automatic feature selection.
  - XGBoost explains more variance but at the cost of higher error (higher RMSE), possibly due to overfitting.
  - Random Forest and Ridge Regression perform well but are slightly less optimal than Lasso.

- Top Features for Predicting House Price:
   - The 20 most important features influencing house prices in this model are:
     - OverallQual, TotalSF, GrLivArea, GarageCars, Age_HouseBlt, TotalBsmtSF, TotRmsAbvGrd, OverallCond, ExterQual, KitchenQual, MSZoning, CentralAir_N, Age_RemodAdd, GarageArea, GarageType, BsmtQualScore
- These features highlight the significance of house quality, total living space, basement area, garage capacity, and renovations in determining property prices.  

## Recommondation:
- Based on this analysis, here is the tips for home buyers:
  * Prioritize quality over just size—a well-built home with high-quality materials may be a better investment.
  * Consider future renovations—homes with good bones but outdated interiors can be upgraded for value appreciation.
  * Check for essential features like garage space, finished basements, and modern kitchens, as they have a direct impact on pricing.
  * Evaluate the neighborhood & zoning regulations to understand long-term investment potential.
  * Older homes may be cheaper, but recent renovations can save future costs.

 🙂 💻
     
