from datasets import load_dataset

dataset = load_dataset('csv', data_files='input.csv')

dataset.save_to_disk('path_to_save')