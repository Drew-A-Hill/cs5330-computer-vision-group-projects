/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  features.h

  Header file for all functions related to computing a set of features for a specified region. 
*/

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>

struct RegionFeatures {
    float centroid_x;
    float centroid_y;
    float angle;                          // axis of least central moment
    cv::RotatedRect oriented_bbox;        // oriented bounding box (4 corners)
    float percent_filled;                 // region area / bounding box area
    float hw_ratio;                       // height / width ratio of bounding box
    double hu_moments[7];
};

RegionFeatures computeFeatures(const cv::Mat &labels, int region_id);

#endif
