#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <filesystem>
#include "features.h"
#include "csv_util.h"
#include <utility>
#include <algorithm>
#include "features_helper.h"
#include "matcher_helper.h"

namespace fs = std::filesystem;

/*
  TEMP NOTE: run ./matcher olympus/{img name.jpg} features.csv {N top imgs to display} action (ie tx for texture)
*/
int main(int argc, char* argv[]){
    std::string imgArg = argv[1];
    cv::Mat targetImg = cv::imread(imgArg);

    std::string csvPath = argv[2];

    // Holds all of the data for each file.
    std::vector<std::vector<float>> data; 

    // Vector containing all of the filenames.
    std::vector<char *> filenames;

    // Retrives all of the filenames and data.
    read_image_data_csv( (char*)csvPath.c_str(), filenames, data);

    int N = std::atoi(argv[3]);
    std::string method = argv[4];

    if(method == "b"){
        std::vector<float> targetImgVector = extractBaselineFeature(targetImg);
        // std::vector<char *> filenames;
        // std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        // read_image_data_csv( (char*)csvPath.c_str(), filenames, data);
        for(int i = 0; i < filenames.size(); i++){
            std::vector<float> &curVec = data[i];
            float distance = ssd(targetImgVector, curVec);
            std::pair<float, std::string> curPair(distance, filenames[i]);
            pairs.push_back(curPair);
        }
        std::sort(pairs.begin(), pairs.end());
        for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }
    }
    else if(method == "h"){
        // FIX: We should put all of the repeats at the top.
        std::vector<float> targetImgHist = extractHist(targetImg);
        // std::vector<char *> filenames;
        // std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        // read_image_data_csv( (char*)csvPath.c_str(), filenames, data);

        for(int i = 0; i < filenames.size(); i++){
            std::vector<float> &curImgHist = data[i];
            float distance = computeHistIntersection(targetImgHist, curImgHist);
            std::pair<float, std::string> curPair(distance, filenames[i]);
            pairs.push_back(curPair);
        }
        std::sort(pairs.begin(), pairs.end());
        for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }
    }
    else if (method == "h3"){
        std::vector<float> targetImgHist = extractHist3x3(targetImg);
        // std::vector<char *> filenames;
        // std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        // read_image_data_csv( (char*)csvPath.c_str(), filenames, data);

        for(int i = 0; i < filenames.size(); i++){
            std::vector<float> &curImgHist = data[i];
            float distance = computeHistIntersection(targetImgHist, curImgHist);
            std::pair<float, std::string> curPair(distance, filenames[i]);
            pairs.push_back(curPair);
        }
        std::sort(pairs.begin(), pairs.end());
        for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }

        // Used for task 4
        // FIXME: Fix this comment
    } else if (method == "tx") {
        // Vector of pairs containing difference between an img and the target img as well as filename.
        std::vector<std::pair<float, std::string>> pairs;

        // Computes the color and texture feature vectors for the target image.
        std::pair<std::vector<float>, std::vector<float>> targetFeatureVectors = computeColorTextureHists(targetImg);
        
        // Declairs a vector of feature vectors for the current image.
        std::pair<std::vector<float>, std::vector<float>> currFeatureVectors;

        for (int i = 0; i < filenames.size(); i++) {
            std::string filepath = std::string("olympus/") + filenames[i];
            cv:Mat currImg = imread(filepath);

            currFeatureVectors = computeColorTextureHists(currImg);

            // Finds the difference for color and texture features.
            float colorDistance = computeHistIntersection(targetFeatureVectors.first, currFeatureVectors.first);
            float textureDistance = computeHistIntersection(targetFeatureVectors.second, currFeatureVectors.second);

            // Finds the combinded weighted difference for each of the features.
            float combinedDistance = (colorDistance * .5) + (textureDistance * .5);

            // Creates a pair and adds the combined distance and the file name of the image compared.
            std::pair<float, std::string> currPair(combinedDistance, filenames[i]);

            pairs.push_back(currPair);
        }

        // Sorts the pairs in order of most to least simmilar.
        std:sort(pairs.begin(), pairs.end());

        // Prints the first N most similar filenames
        for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }
    }
    return 0;
}