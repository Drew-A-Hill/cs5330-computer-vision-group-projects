/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  threshold.cpp

  Implements thresholding functions to seperate dark objects from a white background.
*/
#include "threshold.h"

/*
  const cv::Mat &vs
*/
cv::Mat applyThreshold(const cv::Mat &vs) {
    if (vs.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return cv::Mat();
    }

    // --- Step 1: K-means with K=2 to find the threshold ---

    // Start with two guesses for cluster centers (dark vs bright)
    float mean1 = 0.0f;
    float mean2 = 255.0f;

    for (int iter = 0; iter < 10; iter++) {
        float sum1 = 0, sum2 = 0;
        int count1 = 0, count2 = 0;

        // Sample every 16th pixel for speed
        for (int y = 0; y < vs.rows; y += 16) {
            for (int x = 0; x < vs.cols; x += 16) {
                float pixel = (float)vs.at<uchar>(y, x);

                // Assign pixel to whichever mean is closer
                if (std::abs(pixel - mean1) < std::abs(pixel - mean2)) {
                    sum1 += pixel;
                    count1++;
                } else {
                    sum2 += pixel;
                    count2++;
                }
            }
        }

        // Recompute means from assigned pixels
        if (count1 > 0) mean1 = sum1 / count1;
        if (count2 > 0) mean2 = sum2 / count2;
    }

    // --- Step 2: Threshold is the midpoint between the two means ---
    float threshold = (mean1 + mean2) / 2.0f;

    // --- Step 3: Create binary image using that threshold ---
    // Pixels above threshold = 255 (background), below = 0 (object)
    cv::Mat binary(vs.rows, vs.cols, CV_8UC1);
    for (int y = 0; y < vs.rows; y++) {
        for (int x = 0; x < vs.cols; x++) {
            uchar pixel = vs.at<uchar>(y, x);
            binary.at<uchar>(y, x) = (pixel > threshold) ? 255 : 0;
        }
    }

    return binary;
}

/*
  const cv::Mat &src
*/
cv::Mat extractVS(const cv::Mat &src){
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0, 0);

    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    cv::Mat vs(hsv.rows, hsv.cols, CV_8UC1);
    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            // Get pixel data (3 channels: H, S, V)
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
            
            // Extract S and V values
            uchar saturation = pixel[1];
            uchar value = pixel[2];

            int vs_value = value - saturation;
            if (vs_value < 0){
                vs_value = 0;
            }
            vs.at<uchar>(y, x) = (uchar)vs_value;
        }
    }
    return vs;
}