import pandas as pd
from datasets import Dataset

# Load the dataset from the arrow file
ds = Dataset.from_file("input_file_name.arrow")

# Convert the Dataset object to a pandas DataFrame
df = ds.to_pandas()

# Save the DataFrame to a CSV file
csv_file_path = "output_file_name.csv"
df.to_csv(csv_file_path, index=False)

print(f"Dataset saved to {csv_file_path}")
