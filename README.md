## Usage

Clone the repository, change your present working directory to the cloned directory.
Create folders with names `train_predictions` and `test_outputs` to save model predicted outputs on training and testing images (Not required now as the repo already contains these folders)

```
$ mkdir train_predictions
$ mkdir test_outputs
```

For training the U-Net model and saving weights, run the below command

```
$ python3 main_unet.py
```

To test the U-Net model, calculating accuracies, calculating confusion matrices for training and validation and saving predictions by the model on training, validation and testing images.

```
$ python3 test_unet.py
```

## Note : 

You might get an error `xrange is not defined` while running our code. This error is not due to errors in our code but due to not up to date python package named `libtiff` (some parts of the source code of the package are in python2 and some are in python3) which we used to read the dataset which in which the images are in .tif format. We were not able to use other libraries like openCV or PIL to read the images as they are not properly supporting to read the 4-channel .tif images.

This error can be resolved by editing the source code of the `libtiff` library. 

Go to the file in the source code of the library from where the error arises (the file name will be displayed in the terminal when it is showing the error) and replace all the ```xrange()``` (python2) functions in the file to ```range()``` (python3).

## Pre-trained Model

https://drive.google.com/file/d/1GGwgZGOQeOIwMKPN8WEyIox2CMeHitHx/view?usp=sharing

### Data Processing during training

**The Strided Cropping:**

To have sufficient training data from the given high definition images cropping is required to train the classifier which has about 31M parameters of our U-Net implementation. The crop size of 64x64 we find under-representation of the individual classes and the geometry and continuity of the objects is lost, decreasing the field of view of the convolutions.

Using a cropping window of 128x128 pixels with a stride of 32 resultant of 15887 training and 414 validation images.

**Image Dimensions:**

Before cropping, the dimensions of training images are converted into multiples of stride for convenience during strided cropping.

For the cases where the no. of crops is not the multiple of the image dimensions we initially tried zero padding , we realised that adding padding will add unwanted artefacts in the form of black pixels in training and test images leading to training on false data and image boundary.

Alternatively we have correctly changed the image dimensions by adding extra pixels in the right most side and bottom of the image. So we padded the difference from the left most part of the image to it’s right deficit end and similarly for the top and bottom of the image.

## References

[1] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf), Olaf Ronneberger, Philipp Fischer, and Thomas Brox

[2] [A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review), Sasank Chilamkurthy
