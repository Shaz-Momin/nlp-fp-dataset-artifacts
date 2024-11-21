import json
import string
from datasets import load_dataset

ADD_SENT_EVAL_INPUT = './adv_addSent/eval_output/eval_predictions.jsonl'
ADD_ANY_EVAL_INPUT = './adv_addAny/eval_output/eval_predictions.jsonl'
ADD_COMMON_EVAL_INPUT = './adv_addCommon/eval_output/eval_predictions.jsonl'

ADD_SENT_EVAL_OUTPUT = './error_analysis/addSent_incorrect_pred.jsonl'
ADD_ANY_EVAL_OUTPUT = './error_analysis/addAny_incorrect_pred.jsonl'
ADD_COMMON_EVAL_OUTPUT = './error_analysis/addCommon_incorrect_pred.jsonl'

ADD_SENT_EVAL_INPUT_V2 = './new_eval_outputs/adv_combined_v2/adv_addSent/eval_predictions.jsonl'
ADD_ANY_EVAL_INPUT_V2 = './new_eval_outputs/adv_combined_v2/adv_addAny/eval_predictions.jsonl'
ADD_COMMON_EVAL_INPUT_V2 = './new_eval_outputs/adv_combined_v2/adv_addCommon/eval_predictions.jsonl'

ADD_SENT_EVAL_OUTPUT_V2 = './new_eval_outputs/adv_combined_v2/addSent_incorrect_pred.jsonl'
ADD_ANY_EVAL_OUTPUT_V2 = './new_eval_outputs/adv_combined_v2/addAny_incorrect_pred.jsonl'
ADD_COMMON_EVAL_OUTPUT_V2 = './new_eval_outputs/adv_combined_v2/addCommon_incorrect_pred.jsonl'



# Load the eval_predictions.jsonl file
dataset = load_dataset('json', data_files=ADD_SENT_EVAL_INPUT_V2, split='train')

# Function to remove trailing and leading punctuation from text
def remove_punctuation(text):
    punc = string.punctuation
    return text.strip(punc)

# Function to check if the prediction is correct
def is_prediction_correct(example):
    for answer in example['answers']['text']:
        if remove_punctuation(example['predicted_answer'].lower()) == remove_punctuation(answer.lower()):
            return True
    #return example['predicted_answer'] in example['answers']['text']

# Filter out incorrect predictions
incorrect_predictions = dataset.filter(lambda x: not is_prediction_correct(x))

# Save incorrect predictions to a new file
with open(ADD_SENT_EVAL_OUTPUT_V2, 'w') as f:
    for example in incorrect_predictions:
        f.write(json.dumps(example) + '\n')

print(f"Total incorrect predictions: {len(incorrect_predictions)}")