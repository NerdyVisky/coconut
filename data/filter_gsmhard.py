import json
from transformers import GPT2Tokenizer

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
MAX_TOKENS = 1000  # Set the token limit based on the model

# Define the input and output file paths
input_file = "gsmhardv2.jsonl"  # Change this to your actual file name
output_file = "gsmhardv2_filtered.jsonl"

# List to store the filtered data
data = []

# Read and process the JSONL file
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip())
        text = obj.get("input", "") + " " + obj.get("code", "") + " " + str(obj.get("target", ""))
        token_count = len(tokenizer.encode(text, truncation=False))
                                            
        if token_count <= MAX_TOKENS:
            data.append(obj)

# Write the filtered data back to a JSONL file
with open(output_file, "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry) + "\n")

print(f"Filtered dataset saved as {output_file}")

