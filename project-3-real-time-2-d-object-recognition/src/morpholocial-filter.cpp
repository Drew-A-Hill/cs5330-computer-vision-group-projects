/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  morphological-filter.cpp

  Implements morphological filters on an image to clean it up. 
*/

#include <opencv2/opencv.hpp>

using namespace cv;

/*
  Kernel that can be either 4 connected or 8 connected that searches the target
  pixels neighbors for conflicting binary color. If the target pixel is not 
  the same as the ouput value then it returns. If the target pixel is the same 
  the kernel searches all its neighbors for a conflicting binary color value
  ie if the output is 0 its looking for neighbor 255. If a conflict is found 
  the target pixel is altered.

  cv::Mat &img - reference to the image being filtered.
  int connected - indicates if the filter is a 4 connected or an 8 connected.
  int output - indicates if we are acting on white or black target pixels.
  int size -  the size of the kernel i.e. 3x3 5x5 etc.
  int target[] - location of the target pixel.
  returns a int value indicating if the function was run succesfully.
*/
int kernel(Mat &img, Mat &dst, int connected, int output, int size, int target[]) {
    if (img.empty()) {
        printf("No Image Found");
        return -1;
    }

    // Ensures the output parameter correctly passes 0 or 255.
    if (output != 0 && output != 255) {
        printf("Invalid output value, must be 0 or 255");
        return -1;
    }

    // Checks if target pixel needs to look at neighbors
    if (img.ptr<uchar>(target[0])[target[1]] != (uchar)output) {
        return 1;
    }

    // Checks each neighbor until it finds conflict or checked all neighbors.
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            bool neighbor = true;

            // Enforces the shape of the 4 connected filter to indicate.
            if (connected == 4) {

                // Checks if pixel is target or is not a valid neighbor.
                if (i == j) {
                    neighbor = false;
                } else if (i + j == 0 || i + j == 2) {
                    neighbor = false;
                }
            }

            // Finds start pixel for filter.
            int adj_i = target[0] - 1 + i;
            int adj_j = target[1] - 1 + j;

            int rows = img.rows - 1;
            int cols = img.cols - 1;

            // If looking outside of the frame then skip to next iteration.
            if (adj_i < 0 || adj_j < 0 || adj_i > rows || adj_j > cols) {
                continue;
            }

            uchar *row = img.ptr<uchar>(adj_i);

            // Alters target pixel if neighbor conflicts.
            if (neighbor == true && row[adj_j] != (uchar)output) {
                printf("Flipping pixel at %d -> %d - %d\n", img.ptr<uchar>(target[0])[target[1]], row[adj_j], output);
                dst.ptr<uchar>(target[0])[target[1]] = row[adj_j];
                return 1;
            }
        }
    }
    return 1;
}

/*
  Applys desired erosion or dialiation to an image.

  cv::Mat &img - reference to the image being filtered.
  cv::Mat &dst - reference to a destination for the image.
  int shape - indicates shape of the kernel ie 4 or 8 connected.
  int output - indicates if we are acting on white or black target pixels.
  int size -  the size of the kernel i.e. 3x3 5x5 etc.
*/
int apply_filter(Mat &img, Mat &dst, int shape, int output, int size) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int target[] = {i, j};
            kernel(img, dst, shape, output, size, target);
        }
    }
    return 1;
}

/*
  Erodes the foreground to remove noise then dialates the the forground to
  ensure the foreground remains the same size.

  cv::Mat &img - reference to the image being filtered.
  int shape - indicates shape of the kernel ie 4 or 8 connected.
*/
Mat opening(Mat &img, int shape) {
    Mat temp;
    Mat dst;

    temp = img.clone();
    apply_filter(img, temp, shape, 0, 3);

    dst = temp.clone();
    apply_filter(temp, dst, shape, 255, 5);

    return dst;
}

/*
  Dialates the foreground to remove any holes then erodes the foreground to 
  grow the foreground back to the original size.

  cv::Mat &img - reference to the image being filtered.
  int shape - indicates shape of the kernel ie 4 or 8 connected.
*/
Mat closing(Mat &img, int shape) { 
    Mat temp = img.clone();
    Mat dst;

    apply_filter(img, temp, shape, 255, 5);

    dst = temp.clone();
    apply_filter(temp, dst, shape, 0, 5);        

    return dst;
}