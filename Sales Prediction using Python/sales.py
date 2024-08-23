# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import csv

try:
    # Load your data (replace 'your_data.csv' with your file path)
    # Adjust delimiter and error handling as necessary
    data = pd.read_csv('/content/advertising.csv',delimiter=',', on_bad_lines='skip',quoting=csv.QUOTE_NONE)

    # Display the first few rows of the dataset
    print(data.head())

    # Feature selection: Assuming 'Advertising_Expenditure' and 'Target_Audience' are features
    # and 'Sales' is the target variable
    X = data[['Advertising_Expenditure', 'Target_Audience']]
    y = data['Sales']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Example prediction
    example_data = np.array([[5000, 100]])  # Example feature values
    predicted_sales = model.predict(example_data)
    print(f"Predicted Sales for example data: {predicted_sales[0]}")

except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
