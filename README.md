# House Price Prediction (Machine Learning)

This project predicts house prices using linear regression techniques.

## Overview
The aim of this project is to build a machine learning model that predicts house prices based on features such as area, number of bedrooms, bathrooms, floors, year built, location, condition, and garage availability.

## Dataset
The dataset contains information about houses, including both numerical and categorical features.  
Categorical features such as location, condition, and garage availability are encoded before training the model.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

## Methodology
1. Loaded the house price dataset.
2. Performed data preprocessing and encoding of categorical variables.
3. Split the data into training and testing sets.
4. Trained a Linear Regression model.
5. Evaluated the model using MAE, RMSE, and R² metrics.
6. Compared Linear Regression with Ridge Regression.
7. Selected the final model based on performance.
8. Demonstrated prediction using an example input.

## Results
- The model performance was evaluated using standard regression metrics.
- Linear Regression and Ridge Regression produced similar results.
- An example prediction shows the predicted house price compared with the actual price.

## Conclusion
This project demonstrates an end-to-end machine learning workflow for a regression problem, including data preprocessing, model training, evaluation, and prediction.

## Future Improvements
- Try non-linear models such as Random Forest or Gradient Boosting.
- Improve feature engineering.
- Deploy the model as a web application.

