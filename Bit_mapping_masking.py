import os
import pandas as pd

# Define the number of rows and columns
num_rows = 36
num_cols = 36

# Create a list of lists for the DataFrame
data = []

# Append False rows (first 4 rows)
for _ in range(4):
    data.append([False] * num_cols)

# Append True rows (remaining rows)
for _ in range(num_rows - 4):
    data.append([True] * num_cols)

# Create the DataFrame
df = pd.DataFrame(data)

# Ensure the directory exists
folder_name = 'csv'
os.makedirs(folder_name, exist_ok=True)

# Save the DataFrame to a CSV file in the specified folder
csv_path = os.path.join(folder_name, "map.csv")
df.to_csv(csv_path)  # set index=False if you do not want the row indices in the CSV

