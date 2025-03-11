# House-Price-Prediction
## Regression method using Lasso, RandomFarest, XGBoost, and a detailed EDA
## Table of Content

- [Executive Summary](#Executive-Summary)
- [Introduction](#Introduction)
- [Loading libraries required and reading the data](#Loading-libraries-and-reading-the-data)
- [preproccessing and Exploratory Data Analysis(EDA) ](#preproccessing-and-Exploratory-Data-Analysis-(EDA))
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
- plt.style.use('fivethirtyeight')
- from sklearn.preprocessing import StandardScaler
- from sklearn import metrics
- from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
- from sklearn.preprocessing import LabelEncoder
- from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
- from sklearn.preprocessing import LabelEncoder
- from sklearn.impute import SimpleImputer
- import xgboost as xgb
- import catboost as cb
- from sklearn.decomposition import PCA
- from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
- from sklearn.linear_model import LogisticRegression
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.svm import SVC
- from sklearn import svm
- from sklearn.metrics import confusion_matrix
- from scipy.stats import zscore
- from collections import Counter
- import warnings
- warnings.filterwarnings("ignore")

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
     
