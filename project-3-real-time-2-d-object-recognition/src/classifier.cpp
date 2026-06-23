/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  classifier.cpp
  
  Implements a classificationm of unknown objects using a scaled euclidean distance 
  between the unknown object and labeled data.
*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>


using namespace cv;
using namespace std;

/*
  Computes the standard deviation for each feature for all training data.

  std::vector<std::vector<float>> &data - Training data feature vector.
  Returns a vector of standard deviations.
*/
vector<float> compute_std_dev(vector<vector<float>> &data) {

    // Initializes a Mat with the size of the labled data
    Mat data_mat(data.size(), data[0].size(), CV_32F);

    // Converts from vector of vectors to a Mat
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            data_mat.ptr<float>(i)[j] = data[i][j];
        }
    }

    // Initializes vector holding standard deviation.
    vector<float> std_devs(data[0].size());

    // Computes the standard deviation for each feature and adds it to vector.
    for (int l = 0; l < data[0].size(); l++) {
        Scalar mean;
        Scalar std_dev;

        meanStdDev(data_mat.col(l), mean, std_dev);
        std_devs[l] = (float)std_dev[0];
    }

    return std_devs;
}

/*
  Computes the scaled euclidean distance between unknown features and known features. 

  std::vector<float> &unknown - Feature vector of the object to classify.
  std::vector<float> &known - Feature vector of trainning data.
  std::vector<float> &std_devs - Standard deviation per feature.
  Returns the scaled euclidean distance.
*/
float scaled_euclidean_distance(vector<float> &unknown, vector<float> &known, vector<float> &std_devs) {    
    float sum = 0.0f;

    for (int i = 0; i < unknown.size(); i++) {
        float diff = (unknown[i] - known[i]) / std_devs[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

/*
  Classifies unknown feature vector by fining its nearest neighbor from the training data. 
  
  std::vector float &unknown - Feature vector to be classified.
  std::vector<char*> &labels - Labels used for traning data.
  std::vector<std::vector<float>> & data - Training data feature vector.
  std::vector<float> &std_devs - Standard deviation per feature.
  float threshold - Threshold for distance at which the object will be labeled unknown.
*/
string classify(vector<float> &unknown, vector<char*> &labels, vector<vector<float>> &data, vector<float> &std_dev, float threshold) {
    // Initializes min distance with a very large number.
    float min_dist = numeric_limits<float>::max();

    // Initializes label with unknown
    string selected_label = "Unknown";

    for (int i = 0; i < data.size(); i++) {
        // Finds the distance between the unknown object and each of the objects in training data.
        float dist = scaled_euclidean_distance(unknown, data[i], std_dev);

        if (dist < min_dist) {
            min_dist = dist;
            selected_label = labels[i];
        }
    }

    if (min_dist > threshold) {
        return "Unknown";
    }

    return selected_label;
}