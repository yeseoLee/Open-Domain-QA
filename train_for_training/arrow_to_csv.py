from datasets import Dataset

ds = Dataset.from_file("input_file_name.arrow")

df = ds.to_pandas()

csv_file_path = "output_file_name.csv"
df.to_csv(csv_file_path, index=False)

print(f"Dataset saved to {csv_file_path}")
