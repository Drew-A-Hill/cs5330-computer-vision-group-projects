/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026
  main.cpp

  Entry point for project 3
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "morpholocial-filter.h"
#include "segmenting.h"

int main(int argc, char *argv[]) {
    cv::Mat frame;

    // If an image path is provided, process that single image
    if (argc > 1) {
        frame = cv::imread(argv[1]);
        if (frame.empty()) {
            std::cerr << "Error: could not read image " << argv[1] << std::endl;
            return -1;
        }

        cv::imshow("Original", frame);

        cv::Mat vs = extractVS(frame);

        cv::Mat binary = applyThreshold(vs);                                              
        cv::imshow("Thresholded", binary);

        cv::Mat cleaned = closing(binary, 4);
        cv::imshow("Morphological ", cleaned);

        cv::Mat inverted;
        cv::bitwise_not(cleaned, inverted);

        // Paint border black to disconnect objects from edge regions
        cv::rectangle(inverted, cv::Point(0, 0), cv::Point(inverted.cols - 1, inverted.rows - 1), cv::Scalar(0), 5);

        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;

        int label_count = segment(inverted, 8, labels, stats, centroids);
        int main_label = find_main_region(stats, label_count, inverted.rows, inverted.cols, 200);

        cv::Mat visualize = show_regions(labels, stats, label_count, 200);

        cv::imshow("Segmented Regions", visualize);

        cv::waitKey(0);
        return 0;
    }

    // Otherwise, open webcam for live video
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open webcam" << std::endl;
        return -1;
    }

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Original", frame);
        
        cv::Mat vs = extractVS(frame);
        cv::Mat binary = applyThreshold(vs);                                              
        cv::imshow("Thresholded", binary);

        cv::Mat cleaned = closing(binary, 4);
        cv::imshow("Morphological", cleaned);

        cv::Mat inverted;
        cv::bitwise_not(cleaned, inverted);

        // Paint border black to disconnect objects from edge regions
        cv::rectangle(inverted, cv::Point(0, 0), cv::Point(inverted.cols - 1, inverted.rows - 1), cv::Scalar(0), 5);

        cv::Mat labels, stats, centroids;
        int label_count = segment(inverted, 8, labels, stats, centroids);
        int main_label = find_main_region(stats, label_count, inverted.rows, inverted.cols, 200);

        cv::Mat visualize = show_regions(labels, stats, label_count, 200);
        cv::imshow("Segmented Regions", visualize);

        char key = cv::waitKey(30);
        if (key == 'q') break;
    }

    return 0;
}
