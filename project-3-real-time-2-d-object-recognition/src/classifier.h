/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  classifier.h
  
  The header file for the classification of unknown objects.
*/

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<float> compute_std_dev(vector<vector<float>> &data);
float scaled_euclidean_distance(vector<float> &unknown, vector<float> &known, vector<float> &std_devs);
string classify(vector<float> &unknown, vector<char*> &labels, vector<vector<float>> &data, vector<float> &std_dev, float threshold);

#endif