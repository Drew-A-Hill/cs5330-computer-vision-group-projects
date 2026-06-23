/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  evaluate.cpp

  Runs the full pipeline on a labeled data and builds a confusion 
  matrix comparing true labels to predicted labels.
*/
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include "threshold.h"
#include "morpholocial-filter.h"
#include "segmenting.h"
#include "features.h"
#include "classifier.h"
#include "csv_util.h"

using namespace cv;
using namespace std;

/*
  Runs the classifier against a folder of labeled test images and prints a
  confusion matrix of true vs predicted labels.

  const std::string &test_folder - path to folder of test images, where each filename starts with its true label followed by an underscore.
*/
void run_evaluation(const string &test_folder) {
    // Load the training database so we have known objects to compare against.
    vector<char*> labels_read;
    vector<vector<float>> data;
    read_image_data_csv((char*)"training_data.csv", labels_read, data);

    // Compute standard deviations for the scaled distance metric.
    vector<float> std_dev = compute_std_dev(data);

    // Fixed list of object labels, defines the row and column order of the matrix.
    vector<string> object_labels = {"key", "hat", "pen", "headphones", "phone"};
    int num_classes = object_labels.size();

    // Initializes a confusion matrix using confusion[true_label][predicted_label].
    vector<vector<int>> confusion(num_classes, vector<int>(num_classes, 0));

    // Loop through every file in the test folder.
    for (const auto &entry : filesystem::directory_iterator(test_folder)) {

        // Extract the true label from the filename.
        string filename = entry.path().filename().string();
        string true_label = filename.substr(0, filename.find('_'));

        // Reads the image and skips the file if it failed to load.
        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        // Run the full pipeline.
        cv::Mat vs = extractVS(img);
        cv::Mat binary = applyThreshold(vs);
        cv::Mat cleaned = closing(binary, 4);

        // Invert so the s is foreground for connected components.
        cv::Mat inverted;
        cv::bitwise_not(cleaned, inverted);

        // Paint the border black to disconnect the object from edge regions.
        cv::rectangle(inverted, cv::Point(0, 0), cv::Point(inverted.cols - 1, inverted.rows - 1), cv::Scalar(0), 5);

        // Segment into regions and find the main object region.
        cv::Mat lbl, stats, centroids;
        int label_count = segment(inverted, 8, lbl, stats, centroids);
        int main_label = find_main_region(stats, label_count, inverted.rows, inverted.cols, 200);

        // Skip this image if no valid main region was found.
        if (main_label < 0) continue;

        // Compute features for the main region and classify it.
        RegionFeatures features = computeFeatures(lbl, main_label);
        vector<float> unknown = {features.percent_filled, features.hw_ratio};
        string predicted_label = classify(unknown, labels_read, data, std_dev, 5.0f);

        // Find the matrix indices for the true and predicted labels.
        int true_idx = -1, pred_idx = -1;
        for (int i = 0; i < num_classes; i++) {
            if (object_labels[i] == true_label) true_idx = i;
            if (object_labels[i] == predicted_label) pred_idx = i;
        }

        // Record this result in the confusion matrix if both labels are valid.
        if (true_idx >= 0 && pred_idx >= 0) {
            confusion[true_idx][pred_idx]++;
        }

        // Print the per-image result for reference.
        cout << filename << " | true: " << true_label << " predicted: " << predicted_label << endl;
    }

    // Print the confusion matrix header row (predicted labels across the top).
    cout << "\nConfusion Matrix:\n\t";
    for (auto &lbl : object_labels) cout << lbl << "\t";
    cout << endl;

    // Print each row of the matrix, prefixed with its true label.
    for (int i = 0; i < num_classes; i++) {
        cout << object_labels[i] << "\t";
        for (int j = 0; j < num_classes; j++) {
            cout << confusion[i][j] << "\t";
        }
        cout << endl;
    }
}