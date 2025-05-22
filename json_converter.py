import os
import json
import csv
from tqdm import tqdm  # Optional: progress bar

print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
print("Working directory:", os.getcwd())
script_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(script_dir, 'Data', 'data', 'jsons')
output_csv = os.path.join(script_dir, 'Data/articles.csv')

# Define the fields you want to extract (adjust this to your actual data)
example_file = next(f for f in os.listdir(json_dir) if f.endswith('.json'))
with open(os.path.join(json_dir, example_file), 'r') as f:
    sample_data = json.load(f)
    print("Example JSON structure:\n", json.dumps(sample_data, indent=2))
    # Manually inspect this structure and set the fields below accordingly

# For example, if your JSONs look like:
# {
#   "title": "Article title",
#   "content": "Full text",
#   "bias": 0.5,
#   "source": "CNN"
# }
fields = ["source", "bias", "url", "title", "date", "authors", "content", "content_original", "source_url", "bias_text", "ID"]  # <-- Adjust based on real structure

# Create the CSV and write header
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    # Iterate over all JSON files
    for filename in tqdm(os.listdir(json_dir), desc="Converting JSONs"):
        if filename.endswith('.json'):
            filepath = os.path.join(json_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Filter to keep only the desired fields
                    row = {field: data.get(field, "") for field in fields}
                    writer.writerow(row)
            except Exception as e:
                print(f"Skipping {filename}: {e}")