import requests
import csv
from datetime import datetime, timezone

# Binance API URL
BASE_URL = 'https://api.binance.com/api/v3/klines'

# Parameters for the API request
symbol = 'BTCUSDT'
interval = '1d'  # 1 day candles
limit = 1000  # Maximum data points per request



def date_to_unix(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return int(dt.timestamp() * 1000)

start_time = date_to_unix('2023-09-28') # in Unix ms format

# Function to get historical data from Binance
def get_historical_ohlcv(symbol, interval, start_time, limit):
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }
    
    if start_time:
        params['startTime'] = start_time

    # Make the API request
    response = requests.get(BASE_URL, params=params)
    
    # Check for success
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get data: {response.status_code}, {response.text}")

def save_to_csv(data, filename):
    # Check if the file exists and has content
    file_exists = False
    try:
        with open(filename, 'r') as file:
            file_exists = file.readline() != ''
    except FileNotFoundError:
        pass

    # Open the CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is new or empty
        if not file_exists:
            header = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            writer.writerow(header)
        
        # Append the new data rows
        for row in data:
            timestamp = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)  # Convert timestamp to datetime
            writer.writerow([timestamp, row[1], row[2], row[3], row[4], row[5]])



historical_data = get_historical_ohlcv(symbol, interval, start_time, limit)

save_to_csv(historical_data, 'sui.csv')

print("Data extraction complete and saved to btc_ohlcv_data.csv")
