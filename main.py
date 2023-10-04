import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the historical sales data
data = pd.read_csv('sales_data.csv')

# Split the data into training and test sets
X_train = data.drop(['date', 'sales'], axis=1)
y_train = data['sales']
X_test = X_train.iloc[-10:, :]
y_test = y_train.iloc[-10:]

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = ((y_pred - y_test)**2).mean()

# Print the mean squared error
print('Mean squared error:', mse)
