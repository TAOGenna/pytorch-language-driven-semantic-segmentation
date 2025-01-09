import sys
from pathlib import Path

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import os
print(f"Current working directory: {os.getcwd()}")

file_path = Path('../mseg-api/mseg/dataset_lists/ade20k-150-relabeled/list/train.txt').resolve()
print(f"Resolved path: {file_path}")
print(f"File exists: {file_path.exists()}")

from DPT.utils.dataset import SemData

image_dir = '../data/mseg_dataset/ADEChallengeData2016/'

path_to_imagefiletext = '../mseg-api/mseg/dataset_lists/ade20k-150-relabeled/list/train.txt'

foo = SemData(
    split='train',
    data_root=image_dir,
    data_list=path_to_imagefiletext,
    )

for _ in foo.data_list[:10]:
    print(_)