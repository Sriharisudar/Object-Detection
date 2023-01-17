import os
import splitfolders
#!pip install split-folders

DATA_DIR = "D:\AE_dataset"
CLASSES = os.listdir(DATA_DIR)
print(CLASSES)

img_files = os.listdir(DATA_DIR)
splitfolders.ratio(DATA_DIR, output="D:\AE_split_dataset", seed=1337, ratio=(.7, 0.2,0.1)) 
