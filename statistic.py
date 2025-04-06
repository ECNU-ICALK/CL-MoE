import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="")
args = parser.parse_args()

# Construct txt file path based on task
txt_file_path = "value_counts_" + args.task + '.txt'
print(txt_file_path)

# Use dictionary to map task to output file path
task_to_file = {
    'recognition': "index_recognition.txt",
    'location': "index_location.txt",
    'judge': "index_judge.txt",
    'commonsense': "index_commonsense.txt",
    'count': "index_count.txt",
    'action': "index_action.txt",
    'color': "index_color.txt",
    'type': "index_type.txt",
    'subcategory': "index_subcategory.txt",
    'causal': "index_causal.txt"
}

# Get the output file path based on the task
output_file_path = task_to_file.get(args.task)

# Read txt file and calculate the sum for each index
def calculate_sum_from_txt(file_path):
    # Initialize a dictionary to store the sum for each index
    index_sum_dict = {}

    # Open the file in read mode
    with open(file_path, "r") as txt_file:
        # Read file line by line
        for line in txt_file:
            # Strip newline and split by colon
            index, number = line.strip().split(":")
            index = int(index)
            number = int(number)
            # Add the number to the corresponding index
            if index in index_sum_dict:
                index_sum_dict[index] += number
            else:
                index_sum_dict[index] = number

    return index_sum_dict

# Call the function to calculate sums
result = calculate_sum_from_txt(txt_file_path)

# Sort the dictionary by values in descending order
sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

# Collect the top 8 indices, excluding -1
top_keys = []
for key in sorted_result:
    if key != -1:
        top_keys.append(key)
    if len(top_keys) == 8:
        break

# Write the top indices to the output file
if output_file_path:
    with open(output_file_path, "w") as output_file:
        for key in top_keys:
            output_file.write(f"{key}\n")

# Print the sorted result
print(sorted_result)
