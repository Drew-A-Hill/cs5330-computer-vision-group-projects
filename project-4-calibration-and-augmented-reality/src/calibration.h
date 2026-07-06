/*
  Drew Hill & Abhiram Banda,
  Summer 2026
  CS 5330 Computer Vision

  src/calibration.h
  
  Header file for the calibaration.cpp logic.
*/

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#ifndef CALIBRATION_H
#define CALIBRATION_H

bool detect_corners(Mat &src, Size size, vector<Point2f> &corners);
vector<Vec3f> point_set(Size size);
vector<vector<Vec3f>> point_list(vector<vector<Vec3f>> &point_list, vector<Vec3f> &point_set, bool detected);
vector<vector<Point2f>> corner_list(vector<vector<Point2f>> corner_list, vector<Point2f> corners, bool detected);
pair<Mat, vector<double>> calibrate_camera(Mat &src, vector<vector<Point2f>> corner_list, vector<vector<Vec3f>> point_list);

#endif