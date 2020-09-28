# LEGO Detection
Convolutional Neural Network to detect LEGO Bricks.

## Project Goal

The goal of this project is to correctly classify 14 different types of LEGO bricks in an image with an accuracy of >95% mAP. This is an object detection task. Each image has multiple LEGO's, up to 22 per image. The neural network is trained on *synthetically* generated LEGO images using Blender. The detection on the other hand is on real LEGO images taken by camera. 

The project uses a Mask R-CNN network architecture and is based on this code [Mask R-CNN](https://github.com/matterport/Mask_RCNN). Other network architectures have been tested, such as Retinanet. Also, adding LSTM layers have been tested. However, the results are similar to the bare MASK R-CNN network architecture.

## Project Status

The project is at the following status:
 
- The CNN can detect the LEGO's in a predefined test set of real images to an accuracy of up to 74% mAP.
- This first result is satisfactory. Especially considering that the network was trained on synthetic image data only.
- Detection on synthetic data is very reliable, but not the goal of this project.

This is an object detection result using a real image:

![Test image 0000000002.png](https://github.com/deeepwin/lego-cnn/maskrcnn/datasets/lego/eval/0000000002.png?raw=true "Title")

## Key Challenge

The key challenge at the moment is, that the CNN cannot detect closely neighboring LEGO's on an image reliably.

- Trying to modify the dataset, augmentation, architecture or training process did not help to achieve acceptable accuracy.
- The RPN network has particularly difficulties to locate a LEGO on the image (rois far off), if LEGO's are close to each other.
- First analysis indicates that the neural network (RPN) cannot detect global feature patterns, but instead focues on local patterns

This is an example of how the detection looks like on a image with neighboring LEGO's: 

![Test image 0000000002.png](https://github.com/deeepwin/lego-cnn/maskrcnn/datasets/lego/eval/0000000002.png?raw=true "Title")

I have posted this project, to find interested machine learning enthusiasts, who are willing to contiune the work and solve the current challenge about neighboring LEGo's. Please contribute directly or let me know.

## Project Description

The project is based on the Balloon example provided with Mask R-CNN project. Hence, the folder and data organisation is the same [Balloon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon). You can find the project specific files including the notebooks here:

`../samples/lego`

## Run on Colab

This is the easiest way. 

1. Copy the entire Github project to your Google drive.
2. Copy the dataset into the folder **../datasets/lego**, this might take some time
2. Open a web browser, go to your Google Drive and double-click on the **train_lego_on_colab.ipynb** notebook
3. Click on **Open with Google Colaboratory**
4. Make sure to connect your Google Drive to Colab. This is a button on the left top side.
5. You might need to adjust the paths in the Notebook **/content/drive/My Drive/lego-cnn**
6. Run the notebook

## Data

Training, validation and evaluation data sets must be placed here:

`../datasets/lego`

There are two datasets, all sets contain 1280 training images and 256 validation images. Each data set has the same 8 test images (eval). All images are of size 800x600px.The dataset differ in:

1. Dataset6, contains approximately 9 LEGO's per image which results in 11520 LEGO's to train. All LEGO's are not adjacent. This is the easy dataset.
2. Dataset22, contains approximately 22 LEGO's per image which results in 28160 LEGO's to train. All LEGO's are  adjeaent. This is the hard dataset.

Both datasets are in a zip archive. Just unzip and use either or.



## Train on Local Machine

This project requires TensorFlow 1.x. You need to use the *reqirements.txt* file to install the correct versions of the packages. This is critical. If you use Anaconda, install correct Python environment first through the Anaconda terminal:
  
`conda create --name maskrcnn python=3.5.5
conda activate maskrcnn
conda install GraphViz`

GraphViz is required if you want to plot the model graph. Then install the rest of the packages with pip:

`pip install -r requirements.txt`

In case you face issues in installation, please let me know.


If you train locally make sure you run on GPU with enough memory. Nevertheless, it is often of advangtage to just start the training locally, to check if the configuration is correct and if there is no error, before running it on Colab or Kaggle. To do this, you can use the following python file:

`python train_lego_locally.py`

Uncomment or comment the individual lines to call the main function (tmain) at your wish.


## Evaluation

If you run on Colab, the *train_lego_on_colab* notebook contains already a section to analyse the RPN and a section to run the evaluation (inference). This is the best starting point. If this runs, your all set.

In the folder

`../samples/lego`

you find additional notebooks, that go into more detail. Some are to inspect the network in more detail, such as visualization for the feature maps. Others for inference purposes.















The easiest is to 

