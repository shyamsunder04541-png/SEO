import os
import pandas as pd

# List of CSV file names
files = [
    'Youtube01-Psy.csv',
    'Youtube02-KatyPerry.csv',
    'Youtube03-LMFAO.csv',
    'Youtube04-Eminem.csv',
    'Youtube05-Shakira.csv'
]

# If the files are located inside a folder named 'youtube-spam-collection-v1'
folder_path = 'youtube-spam-collection-v1'

dfs = []
for file in files:
    # Build file path
    path = os.path.join(folder_path, file) if os.path.exists(folder_path) else file
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        dfs.append(df)
    else:
        print(f"Warning: {path} not found.")

if dfs:
    # Combine all DataFrames vertically
    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure output directory exists if saving to 'data/' folder
    os.makedirs('data', exist_ok=True)
    
    # Save combined file to data/comments.csv
    output_path = os.path.join('data', 'comments.csv')
    combined_df.to_csv(output_path, index=False)

    print(f"Successfully combined {len(dfs)} files into {output_path}")
    print(f"Total rows: {len(combined_df)}")
else:
    print("No CSV files were loaded.")