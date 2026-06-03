/*
    Abhiram Banda
    Summer 2026
    CS 5330 Computer Vision

    Include file for features.cpp
*/

#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/opencv.hpp>

// Function declaration (prototype)
std::vector<float> extractBaselineFeature(cv::Mat &image);
float ssd(std::vector<float> &vec1, std::vector<float> &vec2);
std::vector<float> extractHist(cv::Mat &image);
float computeHistIntersection(std::vector<float> &hist1, std::vector<float> &hist2);
std::vector<float> extractHist3x3(cv::Mat &image);

/*
  Computes a histogram for textured images.

  cv::Mat &src Source image to be converted.
  returns std::vector the calculated histogram vector
*/
std::vector<float> computeTexturedHistogram(cv::Mat &src);

#endif // FILTER_H