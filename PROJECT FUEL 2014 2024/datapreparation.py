import pandas as pd

# Load the data (assuming the file path is 'RATES.csv')
df = pd.read_csv('RATES.csv')

# Ensure 'DATES' is in datetime format
df['DATES'] = pd.to_datetime(df['DATES'], format='%d-%m-%Y')

# Sort the data by date to make sure forward fill happens correctly
df = df.sort_values('DATES')

# Forward fill missing values in 'PETROL' and 'DIESEL' columns
df[['PETROL', 'DIESEL']] = df[['PETROL', 'DIESEL']].ffill()

# Apply decimal reduction (round to two decimal places)
df[['PETROL', 'DIESEL']] = df[['PETROL', 'DIESEL']].round(1)

# Set 'DATES' as the index
df.set_index('DATES', inplace=True)

# Save the updated DataFrame back to a CSV file (overwrite or save as a new file)
df.to_csv('U_RATES.csv')  # Saves the updated DataFrame to 'U_RATES.csv'

print("File has been updated, rounded, and saved as 'U_RATES.csv'.")
