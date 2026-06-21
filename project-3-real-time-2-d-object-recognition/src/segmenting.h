/*
Drew Hill & Abhiram Banda
CS 5330 Pattern Recognition & Computer Vision
Summer 2026

segmentation.h

Header file for segmentation.
*/
#include <opencv2/opencv.hpp>

using namespace cv;

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

int segment(Mat &img, int shape, Mat &labeles, Mat &stats, Mat &centroids);
int find_main_region(Mat &stats, int label_count, int rows, int cols, int min);
Mat show_regions(Mat &labels, Mat &stats, int label_count, int min);

#endif