from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load the squad_adversarial dataset
dataset = load_dataset("squad_adversarial", "AddOneSent")

# Filter out examples with "high-conf-turk" in their IDs
filtered_dataset = dataset.filter(lambda example: "high-conf-turk" in example['id'])

# Split the filtered dataset into train and validation sets (80-20 split)
train_test_split = filtered_dataset['validation'].train_test_split(test_size=0.2, seed=42)
dataset_split = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

# Save the train and validation sets to separate files
dataset_split['train'].to_json("./adv_addOneSent/train.jsonl")
dataset_split['validation'].to_json("./adv_addOneSent/validation.jsonl")

print("Train and validation datasets have been saved to train.jsonl and validation.jsonl respectively.")