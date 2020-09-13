# lego-cnn
Convolutional Neural Network to detect LEGO Bricks.

## Project Goal

The goal of this project ist to correctly classify 14 different types of LEGO bricks in an image with an accuracy of >95% mAP). It is a object detection task, in which an image can have multiple LEGO's. The neural network is soley trained on artifically generated LEGO images. The detection on the other hand is on real LEGO images. 

The project uses a MASK R-CNN network architecture and is based on this code https://github.com/matterport/Mask_RCNN. Other network architectures have been used, such as Retinanet, which yielded similar results than MASK R-CNN.

## Project Status

The project is at the following status:
 
- The CNN can detect the LEGO's in a predefined test set with real images to an accuracy of 74% mAP
- The CNN can detect good accuarcy on real LEGO's if they are spaced from each other
- The CNN cannot detect overlapping or neighboring LEGO's very well
