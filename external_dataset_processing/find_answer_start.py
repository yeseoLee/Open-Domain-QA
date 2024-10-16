import pandas as pd

updated_file_path = './input.csv'
data = pd.read_csv(updated_file_path)

def find_start_position(row):
    context = row['context']
    answer = row['answers']
    try:
        start_position = context.find(answer)
        return start_position
    except:
        return -1

data['answer_start'] = data.apply(find_start_position, axis=1)

output_formatted_file_path = './output.csv'
data.to_csv(output_formatted_file_path, index=False)