%tensorflow_version 1.x

import os
import sys

# prepare colab environment and load images
os.chdir('/content/drive/My Drive/Colab/maskrcnn')
os.system('pip install keras==2.2.4') # Must be 2.2.4 (or 2.3.0) otherwise get metrics_tensors error
os.system('python setup.py build_ext --inplace; pip install .')
os.system('pip install keras-resnet')

from samples.lego.lego import main as tmain
from google.cloud import storage

#
# configuration section
#

PATH_DATASET                = os.path.join('datasets', 'lego')

USE_PREV_WEIGHTS			      = False						# if False, set to file name to 'retinanet_152_v1.h5'

#
# main
#

def main():

    # where are we
    print('\Script running in: ' + os.getcwd())

    print("Python version: " + sys.version)
    print("Python install path: " + sys.executable)

    import tensorflow
    print(tensorflow.__version__)
    import keras
    print(keras.__version__)

    # call train function
    if USE_PREV_WEIGHTS:
        #file_weights = tmain([ 'train', '--dataset=' + PATH_DATASET, '--logs=snapshots', '--weights=last','--epochs=4', '--steps-per-epoch=2', '--validation-steps=1'])
        file_weights = tmain([ 'train', '--dataset=' + PATH_DATASET, '--logs=snapshots', '--weights=last','--epochs=4'])
    else:
        #file_weights = tmain([ 'train', '--dataset=' + PATH_DATASET, '--logs=snapshots','--epochs=2', '--steps-per-epoch=3', '--validation-steps=2'])
        file_weights = tmain([ 'train', '--dataset=' + PATH_DATASET, '--logs=snapshots','--epochs=1'])

if __name__ == '__main__':
    main()


