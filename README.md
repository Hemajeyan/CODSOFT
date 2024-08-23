# Titanic Survival Prediction

This project builds a machine learning model to predict whether a passenger survived the Titanic disaster based on available information about the passengers, such as age, gender, ticket class, fare, and other factors.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Model Training](#model-training)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Introduction

The Titanic disaster is one of the most infamous shipwrecks in history. While there was some element of luck in surviving, it appears that some groups of people were more likely to survive than others. This project aims to explore the available dataset of Titanic passengers and develop a model to predict the chances of survival.

## Dataset

The dataset used for this project is the Titanic dataset, which contains detailed information about the passengers. The key features include:

- **PassengerId**: Unique identifier for each passenger
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard the Titanic
- **Parch**: Number of parents or children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number (if known)
- **Embarked**: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- **Survived**: Survival status (0 = No, 1 = Yes) [Target variable]

The dataset can be obtained from [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data).

## Features

- **Data Preprocessing**: 
    - Handling missing data
    - Feature engineering (e.g., extracting titles from names, combining SibSp and Parch into a "FamilySize" feature)
    - One-hot encoding for categorical variables such as gender and embarked location

- **Modeling**:
    - Training machine learning models like Logistic Regression, Random Forest, Decision Tree, and Support Vector Machine (SVM)
    - Cross-validation and hyperparameter tuning

- **Evaluation**:
    - Model evaluation using accuracy, precision, recall, and ROC-AUC score
    - Confusion matrix to analyze model performance

## Model Training

The following models were trained for survival prediction:

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
  
Model selection and performance evaluation were based on cross-validation techniques, with accuracy and ROC-AUC score as the primary metrics.

## Results

- **Accuracy**: X.XX%
- **Precision**: X.XX%
- **Recall**: X.XX%
- **ROC-AUC Score**: X.XX

## Conclusion

This project demonstrates how various machine learning techniques can be applied to predict survival on the Titanic based on passenger attributes. Factors like passenger class, gender, and age played significant roles in determining the survival chances.

## Future Work

- Improve feature engineering by exploring interaction terms between features
- Implement more advanced models like Gradient Boosting or XGBoost
- Explore ensemble methods to combine multiple models for better predictions
- Perform more rigorous hyperparameter tuning using GridSearchCV or RandomizedSearchCV

## References

- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
