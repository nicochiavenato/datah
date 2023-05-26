import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the CSV data into a pandas DataFrame
df = pd.read_csv('data/Product400574.csv')

# Step 2: Prepare the feature matrix X and the target variable y
X = df.drop(['Customer', 'Date', 'Quantity'], axis=1)  # Exclude unnecessary columns
y = df['Quantity']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 5: Predict the target variable for the test data
y_pred = regressor.predict(X_test)

# Step 6: Evaluate the model's performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 7: Prepare new data for prediction
new_data = pd.DataFrame({
    'Unnamed: 0': [0, 1, 2],
    'Region': [1, 2, 3],  # Example values for the features
    'Cat': [2, 1, 3],
    'Route': [1, 2, 3],
    'Product': [400574, 400575, 400576],
    'Season': [3, 3, 2],
    'State': [1, 2, 1],
    'Pack': [2, 2, 1],
    'Manufacturer': [11, 11, 10],
    'Group': [36, 36, 44],
    'Segment': [1, 2, 3],
    'Base Value': [22.82, 26.85, 30.5],
    'Manuf_Value': [26.85, 28.9, 32.1],
})

# Step 8: Make predictions on the new data
predictions = regressor.predict(new_data)

# Step 9: Print the predicted quantities
for i, prediction in enumerate(predictions):
    print(f"Prediction {i+1}: {prediction}")
