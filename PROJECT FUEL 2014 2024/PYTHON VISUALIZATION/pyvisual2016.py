# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('yearly data/2016DATA.csv')

# If Date is not in datetime format, convert it
df['DATES'] = pd.to_datetime(df['DATES'], format='%d-%m-%Y')

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create the line plot
plt.figure(figsize=(10,6))

# Plot PETROL prices
sns.lineplot(x='DATES', y='PETROL', data=df, marker="o", label="Petrol", color="b")

# Plot DIESEL prices
sns.lineplot(x='DATES', y='DIESEL', data=df, marker="o", label="Diesel", color="r")

# Customize the chart
plt.title('Daily Petrol and Diesel Rates Over Time(2014)')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title="Fuel Type")

# Display the plot
plt.tight_layout()
plt.show()
