import os
import sys
from samples.lego.lego import main as tmain


PATH_DATASET        = os.path.join('datasets', 'lego')
IMAGE_PATH          = os.path.join(PATH_DATASET, 'eval', '0000000002.png')
PATH_MODEL_WEIGHTS  = os.path.join('snapshots', 'mask_rcnn_lego_0008.h5')

# Configuration tensorflow 1.15 and keras 2.2.4 works with this code (conda activate C:/Users/Martin/Anaconda3/envs/mrcnn)
# 
# From command line: python.exe samples/lego/lego.py train --dataset=datasets/lego 

# Call train function without any options
file_weights = tmain([ 'train', '--dataset=' + PATH_DATASET, '--logs=snapshots'])

# Call train function with additional options
#tmain([ 'train', '--dataset=' + PATH_DATASET, '--logs=snapshots', '--enable-augmentation', '--epochs=1'])
#tmain([ 'train', '--dataset=' + PATH_DATASET, '--logs=snapshots', '--enable-augmentation', '--weights=last','--epochs=1', '--steps-per-epoch=2', '--validation-steps=1'])
#tmain([ 'evaluate', '--image=' + IMAGE_PATH, '--dataset=' + PATH_DATASET, '--logs=snapshots', '--weights=' + PATH_MODEL_WEIGHTS])
