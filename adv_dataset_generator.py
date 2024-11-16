import json
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

def generate_adv_dataset(dataset_name, split_name, output_dir):
    # Load the squad_adversarial dataset
    dataset = load_dataset(dataset_name, split_name)

    # Filter out examples with "high-conf-turk" in their IDs
    filtered_dataset = dataset.filter(lambda example: "high-conf-turk" in example['id'])

    # Split the filtered dataset into train and validation sets (80-20 split)
    train_test_split = filtered_dataset['validation'].train_test_split(test_size=0.2, seed=42)
    dataset_split = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })

    # Save the train and validation sets to separate files
    dataset_split['train'].to_json(f"{output_dir}/train.jsonl")
    dataset_split['validation'].to_json(f"{output_dir}/validation.jsonl")

    print("Train and validation datasets have been saved to train.jsonl and validation.jsonl respectively.")


# Function to parse the addAny blob and save the examples to a JSONL file
def parse_addAny_blob(blob_path, output_path):
   # Load the JSON file
    with open(blob_path, 'r') as f:
        data = json.load(f)

    # Open the output JSONL file
    with open(output_path, 'w') as f_out:
        for article in data['data']:
            title = article['title']
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    answers = {
                        'text': [answer['text'] for answer in qa['answers']],
                        'answer_start': [answer['answer_start'] for answer in qa['answers']]
                    }
                    entry = {
                        'id': qa['id'],
                        'title': title,
                        'context': context,
                        'question': qa['question'],
                        'answers': answers
                    }
                    
                    # Only write examples that are perturbed (ids contain "adversarial")
                    if "adversarial" in entry['id']:
                        f_out.write(json.dumps(entry) + '\n')

# Split the addAny_output.jsonl file into train and validation sets
def generate_addAny_dataset(input_path, output_dir):
    # Load the dataset from the input file
    dataset = load_dataset('json', data_files=input_path, split='train')

    # Split the dataset into train and validation sets (80-20 split)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_split = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })

    # Save the train and validation sets to separate files
    dataset_split['train'].to_json(f"{output_dir}/train.jsonl")
    dataset_split['validation'].to_json(f"{output_dir}/validation.jsonl")

    print("Train and validation datasets have been saved to train.jsonl and validation.jsonl respectively.")

# TODO: Run functions below as needed (update paths as necessary)
#generate_adv_dataset("squad_adversarial", "AddOneSent", "./adv_addOneSent")
#generate_adv_dataset("squad_adversarial", "AddSent", "./adv_addSent")

#parse_addAny_blob("./all_data.json", "./addAny_output.jsonl")
#generate_addAny_dataset("addAny_output.jsonl", "./adv_addAny")