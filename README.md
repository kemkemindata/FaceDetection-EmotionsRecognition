# Image-Emotions-Classification using CNN

This project is a Convolutional Neural Network (CNN) based model that classifies facial expressions into six categories: `Angry`, `Fear`, `Happy`, `Neutral`, `Sad`, and `Surprise`. The model is trained on a dataset where images are stored in folders corresponding to each emotion, and it is capable of predicting the emotion of a new face image.


## Project Overview

The objective of this project is to classify facial expressions into one of six categories using a CNN model. 
The model is trained on grayscale images of faces, resized to 48x48 pixels. It uses Keras’ `ImageDataGenerator` to load and preprocess the images.
Once the model is trained, it is saved, and users can load it(the model) to classify new images.


Model Architecture
The model is a Convolutional Neural Network (CNN) consisting of:
- Conv2D layers with ReLU activation and MaxPooling.
- A Flatten layer to transform the feature map into a 1D feature vector.
- Dense layers with ReLU and softmax activation for classification into the 6 classes.

## Dataset
The dataset is organized into two main directories:
1. **train**: This folder contains six subfolders, each representing one of the emotions (`angry`, `fear`, `happy`, `neutral`, `sad`, `surprise`). Each subfolder contains images for the respective emotion.
2. **test**: Similar to the train folder, it contains images classified into the same six emotions, used for validation.

**DATASET**
The dataset consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. Each face is based on the emotion shown in the facial expression in one of seven categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

The training set consists of 28,273 images and the test set consists of 7,067 images. 

**Importing the libraries**
The following libraries are imported:
**All libraries used are available in the requirements.txt file**

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import utils

import os

import tensorflow as tf

Further some modules from the Tensorflow library are also imported.




## METHODOLOGY


1. **Install the dependencies**

You can install the required Python packages using pip:

2. **Load the training and test data using ImageDataGenerator.**
Define and compile the CNN model.
Train the model for a specified number of epochs (default: 10).
Save the model as face_classification_model.h5.
Model Parameters: You can adjust the model architecture and training parameters (such as batch size, epochs) in the train.py script.

3. **Training the Model**
Once the dataset is prepared, you can train the model using the provided script. The model will be trained on the train dataset and validated on the test dataset.
First, we select the number of epochs to be 15. The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes through the training dataset. Steps per epoch are calculated by doing a floor division of the number of images in the training generator with the batch size of the training generator. This is repeated for the validation set.


4. **Saving the Model architecture as H5 file**
After training, the model can be saved using Joblib or H5. For the purpose of this project, I used the H5methos to save the model.

5. **Testing the Model**
The model can be tested with new images. You can run the provided script to classify images from the test set or any other folder.

6. **Testing a New Image**
Test the model on a new image:

If you have a new image saved in a folder called new_folder, you can use the following command to test the image:
python predict.py --image new_folder/new_image.jpg
The predict.py script will:

Load the pre-trained model (face_classification_model.h5).
Preprocess the new image.
Predict the emotion class (e.g., Happy, Sad, etc.) of the new image.
Print the predicted emotion on the console.


