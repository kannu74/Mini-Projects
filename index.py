import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def Monthly_sales_grph():
    #6-Grouping sales data by year and month
    monthly_data=sales_data.groupby(['Year','Month'])['Sales'].sum()
    #plotting grouped data
    monthly_data.plot(kind='line', figsize=(10,6))
    plt.title("Monthly Sales")
    plt.xlabel("Months")
    plt.ylabel("Sales")
    plt.show()
    
def Region_Sales_grph():
    region_data=sales_data.groupby('Region')['Sales'].sum()
    region_data.plot(kind='bar',figsize=(8,5),color='skyblue')
    plt.title('Sales by Region')
    plt.xlabel('Region')
    plt.ylabel('Sales')
    plt.show()

def State_Sales_grph():
    state_sales_data=sales_data.groupby('State')['Sales'].sum()
    state_sales_data.plot(kind='pie', 
                            figsize=(6, 6),  # Slightly larger for better display
                            autopct='%1.1f%%',  # Display percentage on the chart
                            startangle=90,  # Start the pie chart at a different angle for better visuals
                            colors=plt.cm.Paired.colors)  # Use a colormap for more vibrant colors
    plt.title('State Sales')
    plt.show()

# 1 - Reading file
sales_data = pd.read_csv("train.csv")

# 2 - Handling Null Values
sales_data = sales_data.fillna(sales_data.mean())

# 3 - Data Pre-Processing (date-time conversion)
sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'])
sales_data['Ship Date'] = pd.to_datetime(sales_data['Ship Date'])

# 4 - Creating Year and Month column for better analysis
sales_data["Year"] = sales_data['Order Date'].dt.year
sales_data["Month"] = sales_data['Order Date'].dt.month

# Encoding categorical variables using one-hot encoding
sales_data = pd.get_dummies(sales_data, columns=['State'])

# For regression, we select relevant features for `x`
x = sales_data[['State_California', 'State_Texas', 'State_New York']]  # Example of dummy-encoded states
y = sales_data['Sales']

# 5 - Train-test splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 6 - Training the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# 7 - Predict sales on the test set
y_pred = model.predict(x_test)

# 8 - Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

# Save predictions to CSV
results = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})
results.to_csv('sales_predictions.csv', index=False)
