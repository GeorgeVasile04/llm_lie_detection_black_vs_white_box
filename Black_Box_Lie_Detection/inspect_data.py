import pandas as pd
import json

# Adjust path if necessary
file_path = r'data/processed_questions/sciq.json'

print(f"Loading {file_path}...")
# read_json automatically handles the column-oriented format
df = pd.read_json(file_path)

print("\n" + "="*80)
print(f"DATASET INFO")
print("="*80)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n" + "="*80)
print("LIST OF ALL COLUMNS")
print("="*80)
# Print columns in sorted order to find them easily
for col in sorted(df.columns):
    print(f"• {col}")

print("\n" + "="*80)
print("PREVIEW (Question, Lie, and Features)")
print("="*80)
# Select a few interesting columns to display
cols_to_show = [
    'question', 
    'false_statement', 
    'text-davinci-003_logprobs_difference_lie'
]

# Filter to only columns that actually exist
cols_to_show = [c for c in cols_to_show if c in df.columns]

# Print first 2 rows
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)
print(df[cols_to_show].head(2))
