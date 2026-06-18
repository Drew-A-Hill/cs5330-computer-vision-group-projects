#include <iostream>
#include <opencv2/opencv.hpp>
#include "threshold.h"

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


        char key = cv::waitKey(30);
        if (key == 'q') break;
    }

    return 0;
}
