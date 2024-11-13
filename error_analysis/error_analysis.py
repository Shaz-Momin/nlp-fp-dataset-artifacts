import json
from datasets import load_dataset

ADD_SENT_EVAL_OUTPUT = './adv_addSent_eval_output/eval_predictions.jsonl'
ADD_ONE_SENT_EVAL_OUTPUT = './adv_addOneSent_eval_output/eval_predictions.jsonl'


# Load the eval_predictions.jsonl file
dataset = load_dataset('json', data_files=ADD_SENT_EVAL_OUTPUT, split='train')

# Function to check if the prediction is correct
def is_prediction_correct(example):
    return example['predicted_answer'] in example['answers']['text']

# Filter out incorrect predictions
incorrect_predictions = dataset.filter(lambda x: not is_prediction_correct(x))

# Save incorrect predictions to a new file
with open('./error_analysis/addSent_incorrect_pred.jsonl', 'w') as f:
    for example in incorrect_predictions:
        f.write(json.dumps(example) + '\n')

print(f"Total incorrect predictions: {len(incorrect_predictions)}")