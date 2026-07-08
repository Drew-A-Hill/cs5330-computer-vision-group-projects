# Project 4: Camera Calibration and Augmented Reality
Drew Hill & Abhiram Banda  
CS 5330 Pattern Recognition and Computer Vision  
Summer 2026
 
## Overview
This project is a camera calibration and augmented reality program. It uses a 9x6 chessboard target to calibrate a camera and display a virtual 3D object on top of it. The program can use a webcam, a mobile phone IP camera stream, a saved video, or a saved image.

## Files  
`main.cpp:` Entry point to the project. Reads the action and input source arguments from the command line and launches the matching runner.  
`action_runner.cpp:` Holds the two run modes. calibration_runner that calibrates the camera by collecting chessboard views, computing the intrinsics, and saving them to file. ar_runner loads the saved intrinsics and overlays the 3D axes and virtual object.  
`action_runner.h:` Header file for action_runner.cpp.  
`calibration.cpp:` Detects chessboard corners, builds the matching 3D point set, stores saved calibration views, and runs calibrateCamera to produce the intrinsics.  
`calibration.h:` Header file for calibration.cpp.  
`ar.cpp:` Reads the saved intrinsics file, computes the pose each frame, and draws the 3D axes and the virtural object.  
`ar.h:` Header file for ar.cpp.

## How to Run
To run the program you need to use `./main {action} {input type} {input info if needed}`. 

### Actions
 `c:` Calibate then use the augmented reality.  
 `ar:` If you have already calibrated and want to just use the augmented reality program.  

### Input
`mobile:` Uses a mobile stream app and requires providing IP address.  
`web:` Uses the users web cam and requires no input info.  
`img:` Uses a still image and requires the name of the image in the folder.  
`video:` Uses a pre recorded video and requires the name of the video file stored in the folder.

### Examples
Calibrates using mobile
```bash
./main c mobile 21.1.0.3
```
Calibrates using web
```bash
./main c web
```
Calibrates using img
```bash
./main c img img.jpeg
```
Calibrates using video
```bash
./main c video vid.mp4
```
Runs augmented reality using mobile
```bash
./main ar mobile 21.1.0.3
```
Runs augmented reality using web
```bash
./main ar web
```
Runs augmented reality using img
```bash
./main ar img img.jpeg
```
Runs augmented reality using video
```bash
./main ar video vid.mp4
```
## Calibration Controls
`0-3:` Set the view rotation.  
`s:` Save the current chessboard detection as a calibration view.  
`c:` Calibrate once 5 or more views have been saved.  
`w:` Write the computed intrinsics and orientation to `./data/intrinsic_parameters.yml`.  
`q:` Quit and print the first corner and corner count of the last detection.
 
## AR Controls
`t:` Toggle hiding the chessboard target by filling it with the sampled surrounding color.  
`r:` Toggle between the virtual house and the virtual rocket.  
`q:` Quit.
 
## Features Program Controls
`h:` Switch to Harris corner detection mode.  
`o:` Switch to ORB feature detection mode.  
`+ / -:` Increase/decrease the Harris threshold or the ORB max feature count.  
`0 - 3:` Rotate the frame.  
`q:` Quit.
 
## Extensions  
### Extension 1
We implemented a feature that hides the checkerboard by projecting the outer board boundary and filling the region with the average color sampled from just outside the boundary. This makes the checkerboard disappear while the virtual objects remain visible, creating a more convincing AR effect. Press 't' to toggle this feature on/off.

 
### Extension 2
We built two virtual objects instead of one. A 3D house and a 3D rocket. The user can toggle between them live with the `r` key.
 
### Extension 3
The system supports loading a static image of the checkerboard and overlaying virtual objects onto it using the saved intrinsic parameters. Run with: ./main ar img <path_to_image>. While the detection and projection work, the results are not as clean as with live video. A perfectly flat, head on checkerboard image provides very little perspective information to estimate depth, causing the virtual object to appear stretched or distorted.

## Travel Days  
We do not need to use any travel days.
