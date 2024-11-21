import os
import json

def combine_files(combined_file):
    folders = ['./adv_addSent', './adv_addAny', './adv_addCommon']
    # Define the paths to the files
    validation_files = [folder + "/" + combined_file + ".jsonl" for folder in folders]

    # Combine the data from all files
    combined_data = []

    for file_path in validation_files:
        with open(file_path, 'r') as file:
            for line in file:
                combined_data.append(json.loads(line))

    # Save the combined data to the new file
    with open("./adv_combined/" + combined_file + ".jsonl", 'w') as outfile:
        for entry in combined_data:
            json.dump(entry, outfile)
            outfile.write('\n')

# Combine the files
#combine_files("train")