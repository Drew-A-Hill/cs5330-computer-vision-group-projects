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
    }
    return 0;
}