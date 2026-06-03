/*
  Drew Hill
  CS5330 Pattern Recognition & Computer Vision 
  Summer 2026

  features_helper.h 

  Header file for the features helper functions.
*/
#ifndef FEATURES_HELPER_H
#define FEATURES_HELPER_H

#include <opencv2/opencv.hpp>

using namespace cv;

/*
  Creates a Sobel Gradient Magnitude textured image from a source image.

  cv::Mat &src Source image to be converted.
  cv::Mat & dst Destination frame to put converted image.
  returns an intiger indicating if runnung the function succeeded or failed.
*/
int textureSobelMagnitude(Mat &src, Mat &dst);

#endif
