/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  segmenting.cpp

  Implements segmentation on a cleaned binary image using OpenCV connectedComponentsWithStats.
  Then filters out regions that are too small or touches the border. Then identifies the main 
  region and produces a colored image indicating regions.
*/

#include <opencv2/opencv.hpp>

using namespace cv;

/*
  Runs the segmentation on a cleaned binary image and returns a lable map and details.

  cv::Mat & img - Cleaned binary image.
  int shape - either 4 or 8 connected.
  cv::Mat &labeles - A map of region labels.
  cv::Mat &stats - Stats about each label.
  cv::Mat &centroids - The centroid per label.
  Returns the number of labels found.
*/
int segment(Mat &img, int shape, Mat &labeles, Mat &stats, Mat &centroids) {
    if (img.empty()) {
        printf("No Image Found");
        return -1;
    }

    // Performs segmentation.
    int label_count = connectedComponentsWithStats(img, labeles, stats, centroids, shape, CV_32S);

    return label_count;
}

/*
  Finds the best object region.
  
  cv::Mat &stats - Stats about each label.
  int label_count - The number of labels.
  int rows - The number of rows in  map.
  int cols - The number of cols in map.
  int min - The smallest size a region can be.
*/
int find_main_region(Mat &stats, int label_count, int rows, int cols, int min) {
    int best_label = -1;
    int best_area = 0;

    for (int i = 0; i < label_count; i++) {
        int area = stats.ptr<int>(i)[CC_STAT_AREA];

        // Checks if a region as atleast the minimum size.
        if (area < min) {
            continue;
        }

        int x = stats.ptr<int>(i)[CC_STAT_LEFT];
        int y = stats.ptr<int>(i)[CC_STAT_TOP];
        int width = stats.ptr<int>(i)[CC_STAT_WIDTH];
        int height = stats.ptr<int>(i)[CC_STAT_HEIGHT];

        // Skips regions that touch the boundary of the image.
        if (x == 0 || y == 0 || x + width == cols || y + height == rows) {
            continue;
        }

        if (area > best_area) {
            best_area = area;
            best_label = i;
        }
    }
    return best_label;
}

/*
  Prodices a visualization showing the different regions.

  cv::Mat &labels - A map of region labels.
  cv::Mat &stats - Stats about each label.
  int label_count - The number of labels.
  int min - The smallest size a region can be.
*/
Mat show_regions(Mat &labels, Mat &stats, int label_count, int min) {
    Mat colored_map = Mat::zeros(labels.size(), CV_8UC3);

    int rows = labels.rows;
    int cols = labels.cols;

    std::vector<Vec3b> colors(label_count);

    // Fixed seed so the same label IDs get consistent colors across frames
    srand(42);

    for (int i = 0; i < label_count; i++) {
        int area = stats.ptr<int>(i)[CC_STAT_AREA];
        int x = stats.ptr<int>(i)[CC_STAT_LEFT];
        int y = stats.ptr<int>(i)[CC_STAT_TOP];
        int width = stats.ptr<int>(i)[CC_STAT_WIDTH];
        int height = stats.ptr<int>(i)[CC_STAT_HEIGHT];

        // Filter out small regions and border-touching regions
        if (area < min || x == 0 || y == 0 || x + width == cols || y + height == rows) {
            colors[i] = Vec3b(0, 0, 0);
        } else {
            colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
        }
    }

    for (int j = 0; j < rows; j++) {
        for (int k = 0; k < cols; k++) {
            int label = labels.at<int>(j, k);

            if (label > 0) {
                colored_map.ptr<Vec3b>(j)[k] = colors[label];
            }
        }
    }
    return colored_map;
}