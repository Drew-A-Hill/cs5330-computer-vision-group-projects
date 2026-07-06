# Project 3: Real Time 2D Object Recognition

Drew Hill & Abhiram Banda  
CS 5330 Pattern Recognition and Computer Vision  
Summer 2026

## Overview

This project is an object recognition program written in C++ with OpenCV. It can use either a webcam or a saved image. The goal is to look at one object in front of the camera, separate it from the background, measure the object, and then predict what object it is.

The program works best with simple objects on a clear background. It can also save new training examples, save test images, and run an evaluation that prints a confusion matrix.

## Files

`main.cpp:` Main program. It handles webcam mode, image mode, training, saving test images, switching classifiers, and evaluation mode.

`threshold.cpp:` Creates the image used for thresholding and separates the object from the background.

`features.cpp:` Finds measurements for the object, such as the center, angle, bounding box, filled area, ratio, and Hu moments.

`features.h:` Header file for the region feature struct and feature function.

`training.cpp:` Saves labeled feature examples to `training_data.csv`.

`evaluation.cpp:` Runs the program on a folder of test images and prints the confusion matrix.

`embedding.cpp:` Handles the ResNet18 classifier. It saves embeddings, loads embeddings, and compares them using cosine distance.

`embedding.h:` Header file for the embedding code.

`utilities.cpp:` Helper code for cropping the object and creating a ResNet18 embedding.

`vision.h:` Header file for the vision helper functions.

## How to Run

To run the program with the webcam:

```bash
./main
```

To run the program on one image:

```bash
./main image.jpg
```

To run evaluation on a folder of test images:

```bash
./main evaluation_mode test_images
```

## Webcam Controls

`q:` Quit the program.

`n:` Save the current object as a training example in `training_data.csv`.

`s:` Save the current frame as a test image.

`e:` Save the current object as a ResNet18 example in `embedding_data.csv`.

`m:` Switch between the normal classifier and the ResNet18 classifier.

## Extensions

### Extension 1

We used ten objects instead of only the required five. The objects were key, pen, carabiner, squeezee, charger, lighter, watch, orange, fork, and pinecone.

### Extension 2

We added extra shape features to the feature based classifier. Along with percent filled and the height to width ratio, we added Hu moments to give the system more information about the shape of each object.

The feature values used were percent filled, height to width ratio, Hu moment 1, and Hu moment 2.

## Travel Days
We would like to use two of out travel days.

