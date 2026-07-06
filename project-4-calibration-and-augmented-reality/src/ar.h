/*
  Drew Hill & Abhiram Banda,
  Summer 2026
  CS 5330 Computer Vision

  src/ar.h

  Header file for the ar.cpp logic.
*/

#ifndef AR_H
#define AR_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

pair<Mat, vector<double>> read_intrinsics(string path, char &orientation);
bool compute_pose(
    Mat &frame, 
    Size size, 
    vector<Vec3f> point_set, 
    vector<Point2f> &corners, 
    pair<Mat, vector<double>> intrinsics,
    Mat &rvec,
    Mat &tvec
    );

void draw_axis(Mat &src, Mat &rvec, Mat &tvec, pair<Mat, vector<double>> &intrinsics);

#endif