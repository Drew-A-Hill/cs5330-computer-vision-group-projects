/*
  Drew Hill
  Pattern Recognition & Computer Vision
  Summer 2026

  macher_helper.cpp

  This file holds the helper methods used within the matcher.cpp file. The use of this file is to
  deculder the main function of matcher.cpp to provide more readable code.
*/
#include <opencv2/opencv.hpp>
#include <vector>
#include "features.h"
#include "features_helper.h"

using namespace cv;
using namespace std;

/*
  Computes the color and textured featured vectors and packages them into a pair.

  cv::Mat the source image being computed.
  returns std::pair<vector<float>, vector<float>> pair with the color and texture vectors for the img.
*/
pair<vector<float>, vector<float>> computeColorTextureHists(Mat &src) {
    Mat texturedImg;

    // Computes a normalized sobel gradient magnitude image for the target image.
    textureSobelMagnitude(src, texturedImg);

    // Computes the feature vector for a textured histogram.
    vector<float> texturedVector = computeTexturedHistogram(texturedImg);

    // Computes the feature vector for the full color histogram.
    vector<float> colorVector = extractBaselineFeature(src);

    // Creates a pair containing the textured feature vector and the color feature vector.
    pair<vector<float>, vector<float>> features(texturedVector, colorVector);

    return features;
}
