import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the CSV data into a pandas DataFrame
selected_data = st.selectbox(
    'Choose the data to predict:',
    ('Product400574.csv', 'Product101810.csv', 'Product404881.csv'))

df = pd.read_csv(f'data/{selected_data}')

# Remove the 'Unnamed: 0' column if present
if 'Unnamed: 0' in df:
    df = df.drop('Unnamed: 0', axis=1)

# Set up Streamlit
st.title('Plots and Linear Regression')

# Plot 1: Line chart of Base Value by Date
st.subheader('Total Base Value by Date')
df['Date'] = pd.to_datetime(df['Date'])
base_value_by_date = df.groupby('Date')['Base Value'].sum()
st.line_chart(base_value_by_date)

# Plot 2: Pie chart of Quantity by Region
st.subheader('Mean Quantity by Region')
quantity_by_region = df.groupby('Region')['Quantity'].mean()
st.bar_chart(quantity_by_region)

# Prepare the feature matrix X and the target variable y
X = df.drop(['Customer', 'Date', 'Quantity'], axis=1)  # Exclude unnecessary columns
y = df['Quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = regressor.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

st.write('Set new inputs for prediction:')

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    region = st.radio('Region:', sorted(df['Region'].unique().tolist()), horizontal=True)
    cat = st.radio('Category:', sorted(df['Cat'].unique().tolist()), horizontal=True)

with c2:
    route = st.radio('Route:', sorted(df['Route'].unique().tolist()), horizontal=True)
    season = st.radio('Season:', sorted(df['Season'].unique().tolist()), horizontal=True)

with c3:
    state = st.radio('State:', sorted(df['State'].unique().tolist()), horizontal=True)
    pack = st.radio('Pack:', sorted(df['Pack'].unique().tolist()), horizontal=True)

with c4:
    manufacturer = st.radio('Manufacture:', sorted(df['Manufacturer'].unique().tolist()), horizontal=True)
    product = st.radio('Product:', sorted(df['Product'].unique().tolist()), horizontal=True)

with c5:
    segment = st.radio('Segment:', sorted(df['Segment'].unique().tolist()), horizontal=True)

group = st.radio('Group:', sorted(df['Group'].unique().tolist()), horizontal=True)
base_value = st.number_input('Base Value:', step=0.01)
manunf_value = st.number_input('Manufactor Value:', step=0.01)


# Prepare new data for prediction
new_data = pd.DataFrame(
    {
        'Region': [region],  # Example values for the features
        'Cat': [cat],
        'Route': [route],
        'Product': [product],
        'Season': [season],
        'State': [state],
        'Pack': [pack],
        'Manufacturer': [manufacturer],
        'Group': [group],
        'Segment': [segment],
        'Base Value': [base_value],
        'Manuf_Value': [manunf_value],
    }
)

# Make predictions on the new data
prediction = regressor.predict(new_data)

# Print the predicted quantities
st.success(f"Quantity Prediction: {prediction[0]} Mean Squared Error: {mse}")
