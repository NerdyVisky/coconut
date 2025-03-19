import json

# Define the input and output file paths
input_file = "gsmhardv2_filtered.jsonl"  # Change this to your actual file name
output_file = "gsmhard_test.json"

# List to store the transformed data
data = []

# Read and process the JSONL file
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip().replace("\n", " "))
        transformed_obj = {
        "question": obj["input"],
        "steps": obj["code"],
        "answer": str(obj["target"])
        }
        data.append(transformed_obj)

# Write to a JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Converted JSONL file saved as {output_file}")
