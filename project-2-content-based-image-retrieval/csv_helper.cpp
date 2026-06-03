#include <cstdio>
#include <cstring>
#include <vector>
#include "opencv2/opencv.hpp"
#include <filesystem>
#include "csv_util.h"
#include "features.h"
#include <set>
namespace fs = std::filesystem;


//
// TODO: We should remove repeat code it appears that all of these condiditonals are doing the exact same thing.
// Is there a reason for that?
// 


int main(int argc, char* argv[]){
    if(argc != 4){
        return 1;
    }
    bool first = true;
    const std::set<std::string> imgExts = { ".png", ".jpg", ".jpeg", ".gif" };
    std::string directoryPath = argv[1];
    fs::path targetDir = directoryPath;
    std::string csvFilename = argv[2];
    std::string featureType = argv[3];

    if(featureType == "b"){
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = extractBaselineFeature(img);

            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    }
    else if(featureType == "h"){
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = extractHist(img);

            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    }
    else if(featureType == "h3"){
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = extractHist3x3(img);

            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    } else if (featureType == "tx") {
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = ;

            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    }
    
    return 0;
}


