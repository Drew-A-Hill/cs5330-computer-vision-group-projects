/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  threshold.h

  Header file for all functions related to performing thresholding. 
*/

#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/opencv.hpp>

cv::Mat applyThreshold(const cv::Mat &vs);
cv::Mat extractVS(const cv::Mat &src);

#endif
