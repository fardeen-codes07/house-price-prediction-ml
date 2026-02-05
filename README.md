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
8. Demonstrated prediction using an example input from the test dataset.

## Results
- The model performance was evaluated using standard regression metrics such as MAE, RMSE, and R².
- Linear Regression and Ridge Regression produced similar results.
- The evaluation showed that the model has limited predictive power on the given dataset, indicating underfitting.

## Output
When the script is executed, the following outputs are displayed in the console:
- Model evaluation metrics (MAE, RMSE, and R² scores)
- Feature coefficients for interpretability
- Train vs Test R² scores
- An example house price prediction along with the actual price for comparison

## How to Run
1. Ensure Python is installed on your system.
2. Install the required libraries: Pandas, NumPy, and Scikit-learn.
3. Update the dataset file path in `main.py` if required.
4. Run the script using:
   ```bash
   python main.py
   ```
   
## Conclusion
This project demonstrates an end-to-end machine learning workflow for a regression problem, including data preprocessing, model training, evaluation, model comparison, and prediction.  
Although the model performance is limited, the project successfully illustrates the complete machine learning pipeline and evaluation process.

## Future Improvements
- Apply non-linear models such as Random Forest or Gradient Boosting.
- Perform better feature engineering.
- Tune hyperparameters for improved performance.

## Dataset Note
The dataset file is not included in this repository. Please update the dataset file path in the code before running the project.

## Author
Fardeen

