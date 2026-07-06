/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  morphological-filter.h

  Header file for all functions related to applying morphological filters. 
*/

#ifndef MORPH_H
#define MORPH_H

#include <opencv2/opencv.hpp>

using namespace cv;

Mat opening(Mat &img, int connected);
Mat closing(Mat &img, int connected); 

#endif