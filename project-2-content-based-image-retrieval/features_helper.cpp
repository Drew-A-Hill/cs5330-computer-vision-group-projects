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
  Helper function that calclulates the value of the target pixel for a sobel 3x3 filter. 
  The filter is used for both Sobel X and Y based on values selected for dir and pass.

  cv::Mat &src source frame.
  cv::Mat &dst destination frame.
  int target[] array contaning the target pixel in {row, col} format.
  int dir direction to apply filter. 0 = horizontal, 1 = vertical.
  int weights[3] weights used for smoothing or differentiating the target pixel.
  int pass the indicates if the filter is on its first or second pass.
  returns int indicating successful or unsuccessful execution.
*/
static int sobelFilter(Mat &src, Mat &dst, int target[2], int dir, int weights[3], int pass) {
    if (src.empty()) {
        printf("Frame is empty/n");
        return - 1;
    }

    // Initializes each of the color channels with a 0 value.
    int blue = 0;
    int green = 0;
    int red = 0;

    int rowLoc;
    int colLoc;

    for (int i = 0; i < 3; i++) {
        // Sets row and column location for horizontal pass.
        if (dir == 0) {
            rowLoc = target[0];
            colLoc = target[1] + i - 1;

        // Sets row and column location for vertical pass.
        } else if (dir == 1) {
            rowLoc = target[0] + i - 1;
            colLoc = target[1];
        }

        // Calculates pixel channel values for first pass.
        if (pass == 0) {
            Vec3b *row = src.ptr<Vec3b>(rowLoc);

            blue += row[colLoc][0] * weights[i];
            green += row[colLoc][1] * weights[i];
            red += row[colLoc][2] * weights[i];

        // Calculates pixel channel values for second pass.
        } else if (pass == 1) {
            Vec3s *row = src.ptr<Vec3s>(rowLoc);

            blue += row[colLoc][0] * weights[i];
            green += row[colLoc][1] * weights[i];
            red += row[colLoc][2] * weights[i];
        }
    }

    // Casts values to be type short for each channel.
    short b = saturate_cast<short>(blue);
    short g = saturate_cast<short>(green);
    short r = saturate_cast<short>(red);

    // Sets target pixel in destination frame.
    dst.ptr<Vec3s>(target[0])[target[1]] = Vec3s(b, g, r);

    return 0;
}

/*
  Converts the source image to 3x3 sobel x filter to find edges in the image. Performs 2 passes 
  using the sobelFilter function to calculate each pixels value. The first pass vertically 
  smoothing each pixel, the second pass to horizontally differentiate the pixel. dst has a type 
  of CV_16SC3 after filter is performed. 

  cv::Mat &src source frame.
  cv::Mat &dst destination frame.
  returns int indicating successful or unsuccessful execution.
*/
static int sobelX3x3(Mat &src, Mat &dst) {
    if (src.empty()) {
        printf("Frame is empty/n");
        return - 1;
    }

    Mat temp;
    // Allocates data to hold frame being teansformed.
    temp.create(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    int weights[3];

    // Sets weights used for smoothing pixels.
    weights[0] = 1;
    weights[1] = 2;
    weights[2] = 1;

    // Vertically smoothes each pixel.
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            int target[2] = {i,j};
            sobelFilter(src, temp, target, 1, weights, 0);
        }
    }

    // Sets weights for differntiating pixels.
    weights[0] = -1;
    weights[1] = 0;
    weights[2] = 1;

    // Horizontially differentiates each pixel.
    for (int i = 1; i < temp.rows - 1; i++) {
        for (int j = 1; j < temp.cols - 1; j++) {
            int target[2] = {i, j}; 
            sobelFilter(temp, dst, target, 0, weights, 1);
        }
    }

    return 0;
}

/*
  Converts a source image to a  3x3 sobel y filter to find edges in the image. Performs 2 passes 
  using the sobelFilter function to calculate each pixels value. The first pass horizontally 
  smoothing each pixel, the second pass to vertically differentiate the pixel. dst has a type of 
  CV_16SC3 after filter is performed. 

  cv::Mat &src source frame.
  cv::Mat &dst destination frame.
  returns int indicating successful or unsuccessful execution.
*/
static int sobelY3x3(Mat &src, Mat &dst) {
    if (src.empty()) {
        printf("Frame is empty/n");
        return - 1;
    }

    Mat temp;
    // Allocates data to hold frame being teansformed.
    temp.create(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    int weights[3];

    // Sets weights used for smoothing pixels.
    weights[0] = 1;
    weights[1] = 2;
    weights[2] = 1;

    // Horizontally smoothes each pixel.
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            int target[2] = {i,j};
            sobelFilter(src, temp, target, 0, weights, 0);
        }
    }

    // Sets weights for differntiating pixels.
    weights[0] = 1;
    weights[1] = 0;
    weights[2] = -1;

    // Vertically differentiates each pixel.
    for (int i = 1; i < temp.rows - 1; i++) {
        for (int j = 1; j < temp.cols - 1; j++) {
            int target[2] = {i, j};
            sobelFilter(temp, dst, target, 1, weights, 1);
        }
    }
    
    return 0;
}

/*
  Converts a source image to a gradient maginitude image using Sobel X and Y filtered imgages. 
  This is done  by calculating the Euclidean norm from a sobel X and sobel Y filtered frame. The 
  value of the Euclidean Norm value for each pixel is then assigned to the destination frame.

  cv::Mat &sx Sobel X filtered frame.
  cv::Mat &sy Sobel Y filtered frame.
  cv::Mat &dst the destination frame.
  returns int indicating successful or unsuccessful execution.
*/
static int magnitude(Mat &sx, Mat &sy, Mat &dst) {
    dst.create(sx.size(), CV_16SC3);
    
    // Checks that neither of the sobel filtered frames are empty.
    if (sx.empty()) {
        printf("SobelX frame is empty\n");

        return -1;
    }

    if (sy.empty()) {
        printf("SobelY frame is empty\n");

        return -1;
    }

    // Fetches pixel for Sobel X and Y filtered frames and calulates L2 Norm for each channel.
    for (int i = 0; i < sx.rows; i++) {
        for (int j = 0; j < sx.cols; j++) {
            Vec3s *x = sx.ptr<Vec3s>(i);
            Vec3s *y = sy.ptr<Vec3s>(i);
            
            // Retrieves pixels.
            Vec3s pixX = x[j];
            Vec3s pixY = y[j];

            // Calculates Norm.
            short lblue = sqrt((pixX[0] * pixX[0]) + (pixY[0] * pixY[0]));
            short lgreen = sqrt((pixX[1] * pixX[1]) + (pixY[1] * pixY[1]));
            short lred = sqrt((pixX[2] * pixX[2]) + (pixY[2] * pixY[2]));

            // Sets destination pixels for each chanel.
            dst.ptr<Vec3s>(i)[j] = Vec3s(lblue, lgreen, lred);
        }
    }
    return 0;
}

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
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);

    // Creates and stores magnitude image.
    magnitude(sx, sy, mag);

    // Convert to CV_8U single-channel for histogram compatibility
    Mat mag8u;
    convertScaleAbs(mag, mag8u);
    cvtColor(mag8u, dst, COLOR_BGR2GRAY);

    return 0;
}