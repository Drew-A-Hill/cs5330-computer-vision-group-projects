/*
  Abhiram Banda, Drew Hill
  Pattern Recognition & Computer Vision
  Summer 2026
*/
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

#ifndef MATCHER_HELP_H
#define MATCHER_HELP_H 

/*
  Computes the color and textured featured vectors and packages them into a pair.

  cv::Mat the source image being computed.
  returns std::pair<vector<float>, vector<float>> pair with the color and texture vectors for the img.
*/
pair<vector<float>, vector<float>> computeColorTextureHists(Mat &src);

#endif