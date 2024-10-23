import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('RATES.csv')
# print(df.columns)

# Convert 'Date' to datetime
df['DATES'] = pd.to_datetime(df['DATES'], dayfirst=True)

# Define election dates for past elections
election_dates = {
    2014: pd.to_datetime('2014-04-10'),
    2016: pd.to_datetime('2016-05-16'),
    2019: pd.to_datetime('2019-04-23'),
    2021: pd.to_datetime('2021-04-06'),
    2024: pd.to_datetime('2024-04-26')
}

# Create a new column 'Days to Election' which indicates days relative to the election date
df['Days_to_Election'] = np.nan

for year, election_date in election_dates.items():
    mask = df['DATES'].dt.year == year
    df.loc[mask, 'Days_to_Election'] = (df['DATES'] - election_date).dt.days

# Filter data to focus on 60 days before and after the election dates
df_filtered = df[(df['Days_to_Election'] >= -60) & (df['Days_to_Election'] <= 60)]

# Create lag features for petrol and diesel prices
df_filtered['PETROL_LAG1'] = df_filtered['PETROL'].shift(1)
df_filtered['DIESEL_LAG1'] = df_filtered['DIESEL'].shift(1)

# Drop any rows with NaN values
df_filtered.dropna(inplace=True)

# Define features and target variables
X = df_filtered[['Days_to_Election', 'PETROL_LAG1', 'DIESEL_LAG1']]
y_petrol = df_filtered['PETROL']
y_diesel = df_filtered['DIESEL']

# Split the data into training and testing sets
X_train, X_test, y_petrol_train, y_petrol_test = train_test_split(X, y_petrol, test_size=0.2, random_state=42)
X_train, X_test, y_diesel_train, y_diesel_test = train_test_split(X, y_diesel, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest models for petrol and diesel price prediction
petrol_model = RandomForestRegressor(n_estimators=100, random_state=42)
petrol_model.fit(X_train_scaled, y_petrol_train)

diesel_model = RandomForestRegressor(n_estimators=100, random_state=42)
diesel_model.fit(X_train_scaled, y_diesel_train)

# Predict prices for 2029 election (use Days_to_Election for 60 days before/after election in 2029)
future_days_to_election = np.arange(-60, 61)  # 60 days before and after election
future_data = pd.DataFrame({
    'Days_to_Election': future_days_to_election,
    'PETROL_LAG1': [np.mean(y_petrol)] * len(future_days_to_election),  # Using average petrol price as lag
    'DIESEL_LAG1': [np.mean(y_diesel)] * len(future_days_to_election)   # Using average diesel price as lag
})

# Standardize the future data
future_data_scaled = scaler.transform(future_data)

# Make predictions for 2029
predicted_petrol_2029 = petrol_model.predict(future_data_scaled)
predicted_diesel_2029 = diesel_model.predict(future_data_scaled)

# Calculate average predicted prices for 2029 election period
average_petrol_2029 = np.mean(predicted_petrol_2029)
average_diesel_2029 = np.mean(predicted_diesel_2029)

print(f"Predicted Average Petrol Price for 2029: {average_petrol_2029}")
print(f"Predicted Average Diesel Price for 2029: {average_diesel_2029}")


# Assuming df has the columns 'DATES', 'PETROL', and 'DIESEL'
plt.figure(figsize=(12, 6))

# Plotting petrol prices
plt.subplot(1, 2, 1)
plt.plot(df['DATES'], df['PETROL'], marker='o', label='Petrol Prices')
plt.title('Historical Petrol Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid()
plt.legend()

# Plotting diesel prices
plt.subplot(1, 2, 2)
plt.plot(df['DATES'], df['DIESEL'], marker='o', label='Diesel Prices', color='orange')
plt.title('Historical Diesel Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()