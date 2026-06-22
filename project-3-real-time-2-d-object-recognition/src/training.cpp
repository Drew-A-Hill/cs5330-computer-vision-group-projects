/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  training.cpp

  Handles saving labeled feature vectors to a CSV file for training the object
  recognition system. During live video, press 'n' to label the current object
  and append its feature vector to training_data.csv.

  CSV format: label,percent_filled,hw_ratio
  Example row: triangle,0.48,0.72

  Usage in main.cpp:
    // When user presses 'n' during webcam loop:
    std::string label;
    std::cout << "Enter object label: ";
    std::cin >> label;
    saveTrainingData(label, features, true);
*/

#include "training.h"
#include <fstream>

/*
  Saves a labeled feature vector to training_data.csv.

  const std::string &label - name of the object (e.g. "triangle", "pen").
  const RegionFeatures &features - computed features for the object.
  bool append - if true, appends to existing file; if false, overwrites.
  Returns 0 on success, -1 on failure.
*/
int saveTrainingData(const std::string &label, const RegionFeatures &features, bool append){
    std::ofstream outfile;
    if (append)
        outfile.open("training_data.csv", std::ios_base::app);
    else
        outfile.open("training_data.csv");

    if (!outfile.is_open()) {
        std::cerr << "Error: could not open training_data.csv" << std::endl;
        return -1;
    }

    outfile << label << "," << features.percent_filled << "," << features.hw_ratio << "\n";
    outfile.close();

    return 0;
}
