#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <filesystem>
#include "features.h"
#include "csv_util.h"
#include <utility>
#include <algorithm>
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

    }
    //Task 4: Texture and Color
    else if (method == "tx") {
        //Compute target image texture features
        cv::Mat texturedImg;
        textureSobelMagnitude(targetImg, texturedImg);
        std::vector<float> targetTextureVec = computeTexturedHistogram(texturedImg);

        //Compute target image color features
        std::vector<float> targetColorVec = extractHist(targetImg);

        int textureSize = targetTextureVec.size();

        std::vector<std::pair<float, std::string>> pairs;

        for(int i = 0; i < filenames.size(); i++){
            //Split the CSV vector into texture and color portions
            std::vector<float> curTexture(data[i].begin(), data[i].begin() + textureSize);
            std::vector<float> curColor(data[i].begin() + textureSize, data[i].end());

            float textureDistance = computeHistIntersection(targetTextureVec, curTexture);
            float colorDistance = computeHistIntersection(targetColorVec, curColor);

            //Weight both equally at 0.5
            float combinedDistance = (textureDistance * 0.5) + (colorDistance * 0.5);

            std::pair<float, std::string> curPair(combinedDistance, filenames[i]);
            pairs.push_back(curPair);
        }
        std::sort(pairs.begin(), pairs.end());
        for(int i = 0; i < N; i++){
            printf("%s\n", pairs[i].second.c_str());
        }
    }
    return 0;
}