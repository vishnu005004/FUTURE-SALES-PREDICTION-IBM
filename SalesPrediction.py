# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('sales.csv')

# Explore the data
print(data.head())  # Display the first few rows of the dataset

# Data preprocessing (you may need to do more extensive preprocessing)
# For example, you can convert date columns to datetime objects
data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')

# Feature engineering (if needed)
# Example: Extract month and year from the date
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Split the data into training and testing sets
X = data[['shop_id', 'item_id', 'month', 'year']]  # Replace with relevant features
y = data['item_cnt_day']  # Replace with your target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_pred = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest Mean Squared Error: {rf_mse}")
print(f"Random Forest R-squared: {rf_r2}")

# Visualize the predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales (Random Forest)')
plt.title('Actual vs. Predicted Sales (Random Forest)')
plt.show()
