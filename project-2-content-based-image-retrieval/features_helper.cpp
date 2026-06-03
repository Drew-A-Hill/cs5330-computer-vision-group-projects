/*
  Drew Hill & Abhiram Banda
  CS5330 Computer Vision & Pattern Recognition
  Summer 2026

  features_helper.cpp
  This is a helper file used for building the features of this project.
*/

#include <opencv2/opencv.hpp>

using namespace cv;

/*
  Creates a Sobel Gradient Magnitude textured image from a source image.

  cv::Mat &src Source image to be converted.
  cv::Mat & dst Destination frame to put converted image.
  returns an intiger indicating if runnung the function succeeded or failed.
*/
int textureSobelMagnitude(Mat &src, Mat &dst) {
    if (src.empty()) {
        printf("Frame is empty\n");
        return - 1;
    }

    // Declares temporary frames
    Mat gray;
    Mat sx;
    Mat sy;
    Mat mag;

    // Converts the source image to a grayscale image.
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Creates and stores a sobel X and Y images.
    Sobel(gray, sx, CV_32F, 1, 0);
    Sobel(gray, sy, CV_16F, 0, 1);

    // Creates and stores magnitude image.
    magnitude(sx, sy, mag);

    normalize(mag, dst, NORM_L2);

    return 1;
}