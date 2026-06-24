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
#include "features.h"
#include "training.h"
#include "classifier.h"
#include "csv_util.h"
#include "evaluation.h"


int main(int argc, char *argv[]) {
    cv::Mat frame;

    // Evaluation mode: ./main --eval test_images/
    if (argc > 2 && std::string(argv[1]) == "--eval") {
        run_evaluation(argv[2]);
        return 0;
    }

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

        if (main_label >= 0) {
            RegionFeatures features = computeFeatures(labels, main_label);

            // Draw the oriented bounding box
            cv::Point2f corners[4];
            features.oriented_bbox.points(corners);
            for (int i = 0; i < 4; i++) {
                cv::line(frame, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }

            // Draw the axis of least central moment
            float len = 100;
            cv::Point2f center(features.centroid_x, features.centroid_y);
            cv::Point2f endpoint(
                center.x + len * cos(features.angle),
                center.y + len * sin(features.angle)
            );
            cv::line(frame, center, endpoint, cv::Scalar(0, 0, 255), 2);

            // Print feature values
            std::cout << "Percent filled: " << features.percent_filled
                      << " | H/W ratio: " << features.hw_ratio
                      << " | Angle: " << features.angle * 180.0f / CV_PI << " deg" << std::endl;

            cv::imshow("Features", frame);
        }

        cv::waitKey(0);
        return 0;
    }

    // Read training data for classificatiom
    std::vector<char*> labels_read;
    std::vector<std::vector<float>> data;
    int read_success = read_image_data_csv((char*)"training_data.csv", labels_read, data);

    std::vector<float> std_dev;
    bool loaded = false;

    if (read_success == 0 && !data.empty()) {
        std_dev = compute_std_dev(data);
        loaded = true;
    } else {
        printf("No training data found, train by pressing n");
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

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat cleaned;
        cv::morphologyEx(binary, cleaned, cv::MORPH_CLOSE, kernel);
        cv::imshow("Morphological", cleaned);

        cv::Mat inverted;
        cv::bitwise_not(cleaned, inverted);

        // Paint border black to disconnect objects from edge regions
        cv::rectangle(inverted, cv::Point(0, 0), cv::Point(inverted.cols - 1, inverted.rows - 1), cv::Scalar(0), 5);

        cv::Mat labels, stats, centroids;
        int label_count = segment(inverted, 8, labels, stats, centroids);
        int main_label = find_main_region(stats, label_count, inverted.rows, inverted.cols, 2000);

        cv::Mat visualize = show_regions(labels, stats, label_count, 200);
        cv::imshow("Segmented Regions", visualize);

        RegionFeatures features;
        if (main_label >= 0) {
            features = computeFeatures(labels, main_label);

            // Draw the oriented bounding box
            cv::Point2f corners[4];
            features.oriented_bbox.points(corners);
            for (int i = 0; i < 4; i++) {
                cv::line(frame, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }

            // Draw the axis of least central moment
            float len = 100;
            cv::Point2f center(features.centroid_x, features.centroid_y);
            cv::Point2f endpoint(
                center.x + len * cos(features.angle),
                center.y + len * sin(features.angle)
            );
            cv::line(frame, center, endpoint, cv::Scalar(0, 0, 255), 2);

            // Classify object
            std::vector<float> unknown = {features.percent_filled, features.hw_ratio, (float)features.hu_moments[0], (float)features.hu_moments[1]};
            std::string selected_label = classify(unknown, labels_read, data, std_dev, 8.0f);

            // Adds the label name to the video stream
            cv::putText(frame, selected_label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Features", frame);
        }

        char key = cv::waitKey(30);
        if (key == 'q') break;

        // Training mode: press 'n' to label and save the current object
        if (key == 'n' && main_label >= 0) {
            std::string label;
            std::cout << "Enter object label: ";
            std::cin >> label;
            saveTrainingData(label, features, true);
            std::cout << "Saved: " << label << std::endl;

            // Reload training data so new samples are used for classification
            labels_read.clear();
            data.clear();
            read_image_data_csv((char*)"training_data.csv", labels_read, data);
            std_dev = compute_std_dev(data);
            loaded = true;
        }

        // Evaluation capture: press 's' to save current frame as a test image
        if (key == 's') {
            static int img_count = 0;
            std::string label;
            std::cout << "Enter true label for test image: ";
            std::cin >> label;
            std::string filename = "test_images/" + label + "_" + std::to_string(img_count++) + ".png";
            cv::imwrite(filename, frame);
            std::cout << "Saved test image: " << filename << std::endl;
        }
    }

    return 0;
}
