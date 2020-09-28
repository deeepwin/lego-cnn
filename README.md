# LEGO Detection
Convolutional Neural Network to detect LEGO Bricks.

## Project Goal

The goal of this project is to correctly classify 14 different types of LEGO bricks in an image with an accuracy of >95% mAP. This is an object detection task. Each image has multiple LEGO's, up to 22 per image. The neural network is trained on *synthetically* generated LEGO images using Blender. The detection on the other hand is on real LEGO images taken by camera. 

The project uses a Mask R-CNN network architecture and is based on this code [Mask R-CNN](https://github.com/matterport/Mask_RCNN). Other network architectures have been tested, such as Retinanet. Also, adding LSTM layers have been tested. However, the results are similar to the bare MASK R-CNN network architecture.

## Project Status

The project is at the following status:
 
- The CNN can detect the LEGO's in a real images to an accuracy of up to 74% mAP.
- This first result is quite, satisfactory, considering that the network was trained on synthetic image data only.
- Detection of LEGO's in a synthetic image is very reliable, but was not the goal of this project.

This is an example of detecting LEGO's in a real image:

![Test image 0000000002.png](https://github.com/deeepwin/lego-cnn/maskrcnn/datasets/lego/eval/0000000002.png?raw=true "Title")

This is an example of detecting LEGO's in a synthetic image, the same type of images the network was trained at:

![Test image 0000000002.png](https://github.com/deeepwin/lego-cnn/maskrcnn/datasets/lego/eval/0000000002.png?raw=true "Title")

The number on the top is the classifier id, that the CNN has detected. The number below is the classifier id ground truth.

## Key Challenge

The key challenge at the moment is, that the CNN cannot detect neighboring LEGO's on an image very reliably.

- Trying to modify the dataset, augmentation, architecture or training process did not help to solve the bad detection accuracy.
- The RPN network has particularly difficulties to locate a LEGO on the image (rois far off), if LEGO's are close to each other.
- First analysis indicates that the neural network (RPN) cannot detect global feature patterns, but instead focues on local patterns.

Typically in a CNN the network builds up more abstract representations of the object as deeper the layers go. However, from the analysis of the detection result, it appears the network focues too much on local patterns. 

This is an example of how the detection looks like on a image with neighboring LEGO's: 

![Test image 0000000002.png](https://github.com/deeepwin/lego-cnn/maskrcnn/datasets/lego/eval/0000000002.png?raw=true "Title")

I have posted this project, to find interested machine learning enthusiasts, who are willing to contiune the work and solve the current challenge about neighboring LEGo's. Please contribute directly or let me know.

## Project Description

This project is based on the Balloon example provided with Mask R-CNN project. Hence, the folder and data organisation is the same. Have a look here to get startet:  [Balloon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon). You can find the project specific files including the notebooks here:


## Data

There are two datasets that you can use. All datasets contain 1280 training images and 256 validation images. Each data set has the same 8 test images (eval folder). All images are of size 800x600px.The datasets differ in the follwwing way:

**Dataset6** - Contains approximately 9 LEGO's per image which results in 11520 LEGO's for training. All LEGO's are not adjacent. This is the easy dataset.

**Dataset22** - Contains approximately 22 LEGO's per image which results in 28160 LEGO's for training. All LEGO's are adjeaent. This is the hard dataset.

Both datasets are in a zip archive. Just unzip and use either or. Place the datasets here in the project folder:

`../datasets/lego/train`

`../datasets/lego/val`

`../datasets/lego/eval`

## Run on Colab

This is the easiest way to run this project.

1. Open a web browser, go to your Google Drive 
2. Copy the entire Github project to your top level Google drive
2. Copy the dataset into the folder **../datasets/lego** as described in the previous section
4. Double-click on the **train_lego_on_colab.ipynb** notebook
5. Click on **Open with Google Colaboratory**
6. Make sure to connect your Google Drive to Colab. This is a button on the left top side
7. You might need to adjust the paths in the notebook **/content/drive/My Drive/lego-cnn**
8. Run the notebook


## Run on Local Machine

To run on your local machine is a bit more tricky. This project requires TensorFlow 1.x. You need to use the *reqirements.txt* file to install the correct versions of the packages. This is critical. If you use Anaconda, install correct Python environment first through the Anaconda terminal:
  
`conda create --name maskrcnn python=3.5.5
conda activate maskrcnn
conda install GraphViz`

GraphViz is required if you want to plot the model graph. Then install the rest of the packages with pip:

`pip install -r requirements.txt`

In case you face issues in installation, please let me know.

If you train locally make sure you run on GPU with enough memory. Nevertheless, it is often of advantage to just start the training locally, to check if the configuration is correct and if there is no error, before running it on Colab or Kaggle. To do this, you can use the following python file:

`python train_lego_locally.py`

Uncomment or comment the individual lines to call the main function (tmain) at your wish.

## Evaluation

If you run on Colab, the **train_lego_on_colab.ipynb** notebook contains already a section to analyse the RPN and a section to run the evaluation (inference). This is the best starting point. If this runs, your all set. You can find in the following folder

`../samples/lego`

additional notebooks, that go into more detail. Some are to inspect the network, such as visualization for the feature maps. Others for inference purposes or checking the datasets.


All the best.













The easiest is to 

