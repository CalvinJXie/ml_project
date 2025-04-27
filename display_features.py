import pandas as pd

# Load the CSV file
data = pd.read_csv('KO_1919-09-06_2025-03-15.csv')

# Display summary statistics for the relevant columns
columns_to_analyze = ['open', 'high', 'low', 'volume']
summary_stats = data[columns_to_analyze].describe()

# Print the summary statistics
print("Summary Statistics for Features:")
print(summary_stats)

# Optionally, check the range of values for each column
print("\nRange of Values for Each Feature:")
for column in columns_to_analyze:
    feature_range = data[column].max() - data[column].min()
    print(f"{column}: {feature_range}")