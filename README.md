# LEGO Detection
Convolutional Neural Network to detect LEGO Bricks.

## Project Goal

The goal of this project is to correctly classify 14 different types of LEGO bricks in an image with an accuracy of >95% mAP. It is a object detection task .Each image chas multiple LEGO's, currently up to 20 per image. The neural network is only trained on synthetically generated LEGO images using Blender. The detection on the other hand is on real LEGO images taken by camera. 

The project uses a MASK R-CNN network architecture and is based on this code [Mask R-CNN](https://github.com/matterport/Mask_RCNN). Other network architectures have been tested, such as Retinanet and adding LSTM layers. However, the results are similar to MASK R-CNN.

## Project Status

The project is at the following status:
 
- The CNN can detect the LEGO's in a predefined test set of real images to an accuracy of 74% mAP
- This result is satisfactory, especially considering that the network was trained only on synthetic image data
- The key challenge at the moment is, that the CNN cannot detect overlapping or neighboring LEGO's reliably
- Trying to modify the data set, augmentation, architecture or training process did not imporive the detection accuracy
- The issue seems that the nerual network cannot detect global feature patterns, but instead focues on local patterns

## Key Challenge

I have posted this project, to find interested machine learning enthusiasts, who are willing to contiune the work and solve the current challenge about neighboring LEGo's.

## Project Description

The project is based on the Baloon example provided with MASK R-CNN. Hence, the folder and data organisation is the same [Balloon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon). You can find the project specific files including the notebooks here:

`../samples/lego`

## Installation

This project requires TensorFlow 1.x. You need to use the *reqirements.txt* file to install the correct versions of the packages. This is critical. If you use Anaconda, install correct Python environment first through the Anaconda terminal:
  
`conda create --name maskrcnn python=3.5.5
conda activate maskrcnn
conda install GraphViz`

GraphViz is required if you want to plot the model graph. Then install the rest of the packages with pip:

`pip install -r requirements.txt`

## Data

Training, validation and evaluation data sets must be placed here:

`../datasets/lego`

There are two datasets, all sets contain 1280 training images and 256 validation images. Each data set has the same 8 test images (eval). All images are of size 800x600px.The dataset differ in:

1. Dataset6, contains approximately 9 LEGO's per image which results in 11520 LEGO's to train. All LEGO's are not adjacent. This is the easy dataset.
2. Dataset22, contains approximately 22 LEGO's per image which results in 28160 LEGO's to train. All LEGO's are  adjeaent. This is the hard dataset.

Both datasets are in a zip archive. Just unzip and use either or.

## Train on Colab

This is the easiest way. Just copy this project on colab server, unzip the dataset and open this notebook on Colab:

`train_lego_on_colab.ipynb`

## Train on Local Machine

If you train locally make sure you run on GPU with enough memory. Nevertheless, it is often of advangtage to just start the training locally, to check if the configuration is correct and if there is no error, before running it on Colab or Kaggle. To do this, you can use the following python file:

`python train_lego_locally.py`

Uncomment or comment the individual lines to call the main function (tmain) at your wish.


## Evaluation

If you run on Colab, the *train_lego_on_colab* notebook contains already a section to analyse the RPN and a section to run the evaluation (inference). This is the best starting point. If this runs, your all set.

In the folder

`../samples/lego`

you find additional notebooks, that go into more detail. Some are to inspect the network in more detail, such as visualization for the feature maps. Others for inference purposes.















The easiest is to 

