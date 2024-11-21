import json
from collections import defaultdict

# input_file = './unperturbed/unperturbed_all.jsonl'
# output_file = './unperturbed/unperturbed_all_deduped.jsonl'

# train_file = './adv_combined/train.jsonl'
# combined_output_file = './adv_combined/train_v2.jsonl'

# unperturbed_count = 0
# max_unperturbed = 800

# with open(combined_output_file, 'w', encoding='utf-8') as combined_outfile:
#     for file in [input_file, train_file]:
#         with open(file, 'r', encoding='utf-8') as infile:
#             for line in infile:
#                 if file == input_file:
#                     if unperturbed_count < max_unperturbed:
#                         combined_outfile.write(line)
#                         unperturbed_count += 1
#                 else:
#                     combined_outfile.write(line)

# print(f"Files combined with {unperturbed_count} unperturbed records. Output saved to {combined_output_file}")

train_file = './adv_combined/train_v2.jsonl'

id_count = defaultdict(int)

with open(train_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        item = json.loads(line)
        item_id = item.get('id')
        if item_id:
            id_count[item_id] += 1

duplicates = {item_id: count for item_id, count in id_count.items() if count > 1}

print(f"Found {len(duplicates)} duplicate items.")
for item_id, count in duplicates.items():
    print(f"ID: {item_id}, Count: {count}")