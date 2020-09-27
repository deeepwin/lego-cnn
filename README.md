# lego-cnn
Convolutional Neural Network to detect LEGO Bricks.

## Project Goal

The goal of this project is to correctly classify 14 different types of LEGO bricks in an image with an accuracy of >95% mAP. It is a object detection task .Each image chas multiple LEGO's, currently up to 20 per image. The neural network is only trained on synthetically generated LEGO images using Blender. The detection on the other hand is on real LEGO images taken by camera. 

The project uses a MASK R-CNN network architecture and is based on this code https://github.com/matterport/Mask_RCNN. Other network architectures have been tested, such as Retinanet and adding LSTM layers. However, the results are similar to MASK R-CNN.

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

The project is based on the Baloon example provided with MASK R-CNN. Hence, the folder and data organisation is the same (https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)

### Data



