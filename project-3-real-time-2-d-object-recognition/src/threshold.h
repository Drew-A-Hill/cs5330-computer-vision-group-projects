#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/opencv.hpp>

// TODO: You'll add your thresholding function declarations here
cv::Mat applyThreshold(const cv::Mat &vs);

cv::Mat extractVS(const cv::Mat &src);

#endif
