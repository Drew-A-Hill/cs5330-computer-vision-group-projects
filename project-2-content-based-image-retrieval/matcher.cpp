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

namespace fs = std::filesystem;

int main(int argc, char* argv[]){
    std::string imgArg = argv[1];
    cv::Mat targetImg = cv::imread(imgArg);

    std::string csvPath = argv[2];

    int N = std::atoi(argv[3]);
    std::string method = argv[4];

    if(method == "b"){
        std::vector<float> targetImgVector = extractBaselineFeature(targetImg);
        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        read_image_data_csv( (char*)csvPath.c_str(), filenames, data);
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

        std::vector<float> targetImgHist = extractHist(targetImg);
        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        read_image_data_csv( (char*)csvPath.c_str(), filenames, data);

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
        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        read_image_data_csv( (char*)csvPath.c_str(), filenames, data);

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
        
        cv::Mat texturedTargetImg;

        // Computes a normalized sobel gradient magnitude image for the target image.
        textureSobelMagnitude(targetImg, texturedTargetImg);

        // Computes the feature vector for a textured histogram.
        std::vector<float> texturedFeatureVector = computeTexturedHistogram(texturedTargetImg);

        // Computes the feature vector for the full color histogram.
        std::vector<float> colorFeatureVector = extractBaselineFeature(targetImg);

        std::vector<float> distanceTextured = computeHistIntersection()





    }
    return 0;
}