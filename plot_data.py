import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
# Replace 'your_file.csv' with the actual file path
data = pd.read_csv('KO_1919-09-06_2025-03-15.csv')

# Ensure the 'Date' column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['close'], label='Close Price', color='blue')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Close Price Over Time')
plt.legend()
plt.grid()

# Show the plot
plt.show()