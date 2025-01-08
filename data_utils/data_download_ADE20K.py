# What do we need in terms of data? 
# COCO
# ADE20K

# Say they are at ../COCO/ and ../ADE20K/
# We need: original images, annotated images, classes/tags (json?)
import dataset_tools as dtools
import os


ADE20K_dir = '../../ADE20K_new/'
if os.path.exists(ADE20K_dir) is False and os.path.isdir(ADE20K_dir) is False:
    dtools.download(dataset='ADE20K', dst_dir=ADE20K_dir)
else:
    print(f'{ADE20K_dir} exists')


