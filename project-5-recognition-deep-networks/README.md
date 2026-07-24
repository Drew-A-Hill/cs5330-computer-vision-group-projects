# Project 5: Recognition using Deep Networks

Drew Hill & Abhiram Banda  
CS5330 Pattern Recognition and Computer Vision  
Summer 2026

## Overview

This project is a digit recognition program written in Python with PyTorch and OpenCV. It trains a convolutional neural network on the MNIST dataset. The trained network is then used to read handwritten numbers from photos and from a live video stream.

The project also looks inside the network to see what the filters learned. It reuses the trained network to recognize Greek letters and runs an experiment on the Fashion MNIST dataset to compare different network settings.

## Files

`model.py:` Defines the network and trains it on MNIST. Saves the weights to model_weights.pth.

`load_model.py:` Loads the trained network and runs it on the first ten test images. Prints the ten output values, the index of the biggest one, and the correct label. Then plots the predictions.

`network.py:` Prints the network layout and plots the ten filters from the first convolution layer.

`task2b.py:` Applies each of the ten first layer filters to a training image. Plots the filter next to the result.

`task3.py:` Transfer learning. Freezes the trained network and replaces the last layer with one that has three outputs. Trains it on alpha, beta, and gamma. Saves the weights to `greek_model_weights.pth`.

`experiment.py:` Runs the design experiment on Fashion MNIST. Changes the number of filters, the dropout rate, and the batch size. Saves the results to experiment_results.csv.

`transformer.py:` A vision transformer trained on MNIST. Splits each image into patches and uses a transformer encoder instead of convolution layers.

`handwritten.py:` Reads photos of handwritten numbers. Converts them to match the MNIST format and runs them through the trained network.

`network_diagram.py:` Draws the diagram of the network architecture.

`ext_pretrained.py:` Extension. Loads the pretrained ResNet18 and shows the filters from its first two convolution layers.

`ext_live_video.py:` Extension. Reads handwritten numbers from a live video stream instead of saved photos.

## Additional Greek Letter Examples

Our own alpha, beta, and gamma images used to test the trained network: https://drive.google.com/drive/folders/19ffNWAbgyGvWJPAH4P_wcXL0Hd9H59h2?usp=sharing

## How to Run

The scripts save their figures into a results folder. That folder needs to exist before running them. The MNIST and Fashion MNIST datasets download themselves into a data folder the first time they are used.

To train the network and save the weights:

```bash
python model.py
```

To run the trained network on the test set:

```bash
python load_model.py
```

To look at the filters of the first layer:

```bash
python network.py
python task2b.py
```

To train the Greek letter network:

```bash
python task3.py
```

To run the experiment on Fashion MNIST:

```bash
python experiment.py
```

To train the transformer:

```bash
python transformer.py
```

To read the handwritten number photos:

```bash
python handwritten.py
```

To look at the pretrained ResNet18 filters:

```bash
python ext_pretrained.py
```

To read numbers from the webcam:

```bash
python ext_live_video.py
```

To read numbers from a phone camera pass the address of the phone for example:

```bash
python ext_live_video.py 192.168.1.100
```

## Video Controls

`q:` Quit the program and print how many of each number were saved.

`s:` Save the current guess.

## Extensions

### Extension 1

We loaded the pretrained ResNet18 network from torchvision and looked at the filters in its first two convolution layers. They fall into two groups. Some are edge detectors at different angles. The rest are color blobs that react to changes between colors. Both groups look much cleaner than the filters our own network learned. We also ran the first layer filters over one of our handwritten number photos. The filters were trained only on photographs but they still pick out the edges of a digit they have never seen.

### Extension 2

We added a live video mode that reads handwritten numbers from a webcam or a phone camera instead of from saved photos. A green box is drawn in the middle of the frame to show where to hold the number.Each frame is cropped to that box and converted the same way as the saved photos, then run through the trained network. The guess is written above the box. Blank frames are skipped so the program does not guess when there is nothing to read. Numbers can be saved while the program is running. When the program exits it prints a count of everything that was saved.

## Acknowledgement
We used AI to debug code, understand documentation, and for grammar writing aid.

## Travel Days

We would like to use one of our travel days.