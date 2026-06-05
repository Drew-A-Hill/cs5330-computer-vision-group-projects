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


/*  
    This program holds the logic for writing to a csv file after taking in args. Based on the featureType provided, it will 
    call the correct method from features.cpp on each image and write the vector to the csv
*/
int main(int argc, char* argv[]){
    if(argc != 4){
        return 1;
    }
    bool first = true;
    //set of file extensions which we use to check if a path is really an image
    const std::set<std::string> imgExts = { ".png", ".jpg", ".jpeg", ".gif" };
    //Arg1: the path to the directory of images
    std::string directoryPath = argv[1];
    fs::path targetDir = directoryPath;
    //Arg2: The name of the csv file which the program will create
    std::string csvFilename = argv[2];
    //Arg3: The feature type which dictates which logic to use when extracting the vector from an image
    std::string featureType = argv[3];

    //First task: Baseline Extraction
    if(featureType == "b"){
        //loops through the image directory
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();
            //check if the path is an image
            if (!imgExts.count(ext)) {
                continue;
            }
            //extracts the vector from the current image
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = extractBaselineFeature(img);

            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    }
    //Second task: Histogram Matching
    else if(featureType == "h"){
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            //Extracts vector/hist from the image
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = extractHist(img);

            //Adds the filename and its vector to the csv
            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    }
    //Second task: Histogram Matching with 3d matrix
    else if(featureType == "h3"){
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            //Extracts vector/hist from the image
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = extractHist3x3(img);

            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    } 
    //Task 3: Multi Histogram Matching
    else if(featureType == "mh"){
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            //Extracts vector/hist from the image
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = extractMultipleHistQuadrants(img);

            //Adds the filename and its vector to the csv
            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    }
    else if(featureType == "mh2"){
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            //Extracts vector/hist from the image
            cv::Mat img = cv::imread(entry.path().string());
            std::vector<float> vec = extractMultipleHistFullMiddle(img);

            //Adds the filename and its vector to the csv
            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    }
    //Task 4: Textured Features
    else if (featureType == "tx") {
        for (const auto& entry : fs::directory_iterator(targetDir)) {
            std::string ext = entry.path().extension().string();

            if (!imgExts.count(ext)) {
                continue;
            }
            cv::Mat img = cv::imread(entry.path().string());

            //Compute texture features (Sobel magnitude histogram)
            cv::Mat texturedImg;
            textureSobelMagnitude(img, texturedImg);
            std::vector<float> textureVec = computeTexturedHistogram(texturedImg);

            //Compute color features (whole image color histogram)
            std::vector<float> colorVec = extractHist(img);

            //Concatenate texture + color into a single vector
            std::vector<float> vec;
            vec.insert(vec.end(), textureVec.begin(), textureVec.end());
            vec.insert(vec.end(), colorVec.begin(), colorVec.end());

            //Adds the filename and its vector to the csv
            append_image_data_csv((char*)csvFilename.c_str(), (char*)entry.path().filename().string().c_str(),vec, first ? 1:0);
            first = false;
        }
    }

    return 0;
}


