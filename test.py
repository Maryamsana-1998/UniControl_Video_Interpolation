from src.train.util import load_caption_dict
import os
import glob 
from pathlib import Path
import shutil

# Load all sequence folders
sequences = glob.glob('/data2/local_datasets/vimeo_sequences/*/*')
ids = []
for seq in sequences:
    dir_part = Path(seq).parent.name.zfill(5)
    file_part = Path(seq).name  # changed .stem to .name to preserve full name
    seq_id = f"{dir_part}_{file_part}"
    ids.append(seq_id)

ids = set(ids)

# Load available captions
caps = load_caption_dict('data/final_captions.txt')
caps = set(caps.keys())

# Compute missing IDs
missing = sorted(list(ids - caps))
print(f"Total Sequences: {len(ids)}")
print(f"Available Captions: {len(caps)}")
print(f"Missing Captions: {len(missing)}")
print("Sample missing:", missing[:5])

# missing_folders = []

# for mis in missing:
#     sp = mis.split('_')
#     if sp[0].startswith("0"):
#         sp[0] = sp[0][1:]  # Remove only one leading zero

#     folder_path = Path(f"/data2/local_datasets/vimeo_sequences/{sp[0]}/{sp[1]+'_'+sp[2]}")
#     if folder_path.exists():
#         print(f"Deleting: {folder_path}")
#         shutil.rmtree(folder_path)
#     else:
#         print(f"Not found: {folder_path}")
