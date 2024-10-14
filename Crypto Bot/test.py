import pandas as pd
from datetime import datetime, timedelta

# Load the data from the CSV file
filename = 'sui.csv'

# Read the CSV into a pandas DataFrame
data = pd.read_csv(filename)

# Convert the 'timestamp' column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Format the 'timestamp' to only show the date (YYYY-MM-DD)
data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d')

# Sort the data by date in case it's not sorted
data = data.sort_values(by='timestamp')

# Define the split point (last 30 days)
last_date = data['timestamp'].max()  # Find the latest date in the dataset
split_date = pd.to_datetime(last_date) - timedelta(days=30)

# Split the data into training and test sets
train_data = data[pd.to_datetime(data['timestamp']) < split_date]
test_data = data[pd.to_datetime(data['timestamp']) >= split_date]

# Now we append the last 30 days of the training data to the beginning of the test set
# So we have a rolling window of the last 30 days to predict the first day of test data
rolling_window_data = pd.concat([train_data.tail(30), test_data])

# Save the training data
train_data.to_csv('btc_ohlcv_train.csv', index=False)

# Save the rolling window test set (which includes the last 30 days of training)
rolling_window_data.to_csv('btc_ohlcv_test_with_train_window.csv', index=False)

print(f"Data successfully split into training and test sets.")
print(f"Training data saved to 'btc_ohlcv_train.csv'.")
print(f"Test data with rolling window saved to 'btc_ohlcv_test_with_train_window.csv'.")
