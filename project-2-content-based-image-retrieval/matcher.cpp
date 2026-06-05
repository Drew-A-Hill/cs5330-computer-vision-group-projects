/*
  Drew Hill
  Project 2
  Pattern Recognition & Computer Vision
  Summer 2026

  matcher.cpp

  This file is used to compare the target image to the other images in the database and provide an
  ordered vector of images that are most similar to the target image.
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <filesystem>
#include "features.h"
#include "csv_util.h"
#include <utility>
#include <algorithm>
#include <unordered_map>

namespace fs = std::filesystem;

/*
  Task 1: Baseline Matching
  This method computes the baseline feature diffence between the target image and each of the other
  images in the data base.

  cv::Mat targetImg the target image.

*/
std::vector<std::pair<float, std::string>> baselineMatching(
    cv::Mat targetImg, std::vector<char *> filenames, 
    std::vector<std::vector<float>> data) {

    // Computes the features for the target image returned as a vector.
    std::vector<float> targetImgVector = extractBaselineFeature(targetImg);

    // Declares variable to hold vector of pairs containing the distance to target image and img filename
    std::vector<std::pair<float, std::string>> pairs;

    // Computes the distance of target img to each other img and stores them in a vector of pairs.
    for(int i = 0; i < filenames.size(); i++) {
        std::vector<float> &curVec = data[i];
        float distance = ssd(targetImgVector, curVec);

        std::pair<float, std::string> curPair(distance, filenames[i]);
        pairs.push_back(curPair);
    }

    // Sorts vector of distance pairs
    std::sort(pairs.begin(), pairs.end());

    return pairs;
}

/*
  Task 2: Histogram Matching
*/
std::vector<std::pair<float, std::string>>histMatching(cv::Mat targetImg, std::vector<char *> filenames, std::vector<std::vector<float>> data) {

    // Computes a color histogram 
    std::vector<float> targetImgHist = extractHist(targetImg);

    // Declares variable to hold vector of pairs containing the distance to target image and img filename
    std::vector<std::pair<float, std::string>> pairs;

    // Computes the distance of target img to each other img and stores them in a vector of pairs.
    for (int i = 0; i < filenames.size(); i++) {
        std::vector<float> &curImgHist = data[i];

        // Computes the distince between target and current image.
        float distance = computeHistIntersection(targetImgHist, curImgHist);

        std::pair<float, std::string> curPair(distance, filenames[i]);
        pairs.push_back(curPair);
    }

    // Sorts vector of distance pairs
    std::sort(pairs.begin(), pairs.end());

    return pairs;
}

/*
  Task 2: Histogram matching 3x3 histogram.
*/
std::vector<std::pair<float, std::string>> hist3x3Matching(cv::Mat targetImg, std::vector<char *> filenames, std::vector<std::vector<float>> data) {
    // Computes 3x3 histogram.
    std::vector<float> targetImgHist = extractHist3x3(targetImg);
    
    // Declares variable to hold vector of pairs containing the distance to target image and img filename.
    std::vector<std::pair<float, std::string>> pairs;
    
    for (int i = 0; i < filenames.size(); i++) {
        std::vector<float> &curImgHist = data[i];

        // Computes the distince between target and current image
        float distance = computeHistIntersection(targetImgHist, curImgHist);

        std::pair<float, std::string> curPair(distance, filenames[i]);
        pairs.push_back(curPair);
            
    }

    // Sorts vector of distance pairs
    std::sort(pairs.begin(), pairs.end());

    return pairs;
}

/*
  Task 3: Multiple Histogram Matching
*/
std::vector<std::pair<float, std::string>> multiHistMatching(cv::Mat targetImg, std::vector<char *> filenames, std::vector<std::vector<float>> data) {
    //holds the vector for the targetImage whcih we'll use to compare with each image
    std::vector<float> targetImgHist = extractMultipleHistQuadrants(targetImg);

    //Compute chunksize because we broke the image into 4 different images
    int chunkSize = targetImgHist.size() / 4;

    //Uses chunkSize to break the big vector into its 4 separate vectors represingting the portions of the image we chose
    std::vector<float> targetTop(targetImgHist.begin(), targetImgHist.begin() + chunkSize);                              
    std::vector<float> targetBottom(targetImgHist.begin() + chunkSize, targetImgHist.begin() + 2*chunkSize);                                    
    std::vector<float> targetLeft(targetImgHist.begin() + 2*chunkSize, targetImgHist.begin() + 3*chunkSize);                                    
    std::vector<float> targetRight(targetImgHist.begin() + 3*chunkSize, targetImgHist.end());     

    // Declares variable to hold vector of pairs containing the distance to target image and img filename.
    std::vector<std::pair<float, std::string>> pairs;

    //loops through each image and its vector
    for (int i = 0; i < filenames.size(); i++) {
        std::vector<float> &curImgHist = data[i];
        
        //Does the same for each image and its vector in the loop so we can correctly compare it to the targetImage
        std::vector<float> curTop(curImgHist.begin(), curImgHist.begin() + chunkSize);                              
        std::vector<float> curBottom(curImgHist.begin() + chunkSize, curImgHist.begin() + 2*chunkSize);                                    
        std::vector<float> curLeft(curImgHist.begin() + 2*chunkSize, curImgHist.begin() + 3*chunkSize);                                    
        std::vector<float> curRight(curImgHist.begin() + 3*chunkSize, curImgHist.end());

        //Give skewed preference to the Bottom half of the image - this is for fun and just testing. the next method is also part
        //this task 3 wehere we use a better measure
        float distance = 0.1 * computeHistIntersection(targetTop, curTop) 
            + 0.80 * computeHistIntersection(targetBottom, curBottom)
            + 0.05 * computeHistIntersection(targetLeft, curLeft)     
            + 0.05 * computeHistIntersection(targetRight, curRight);

        std::pair<float, std::string> curPair(distance, filenames[i]);
        pairs.push_back(curPair);
    }

    // Sorts vector of distance pairs
    std::sort(pairs.begin(), pairs.end());

    return pairs;
}

/*
  Task 3: Multi Histogram Matching Part 2.

*/
std::vector<std::pair<float, std::string>> multiHistFullMiddleMatch(
    cv::Mat targetImg, 
    std::vector<char *> filenames, 
    std::vector<std::vector<float>> data) {

    //Similar to the previous method except this measure breaks the image into the full image and middle of the image
    std::vector<float> targetImgHist = extractMultipleHistFullMiddle(targetImg);
    //Thats why we break into 2 instead of 4
    int chunkSize = targetImgHist.size() / 2;

    std::vector<float> targetFull(targetImgHist.begin(), targetImgHist.begin() + chunkSize);                              
    std::vector<float> targetMiddle(targetImgHist.begin() + chunkSize, targetImgHist.begin() + 2*chunkSize);                                      

    std::vector<std::pair<float, std::string>> pairs;
    for(int i = 0; i < filenames.size(); i++){
        std::vector<float> &curImgHist = data[i];
        //Same logic for each image and its vector
        std::vector<float> curFull(curImgHist.begin(), curImgHist.begin() + chunkSize);                              
        std::vector<float> curMiddle(curImgHist.begin() + chunkSize, curImgHist.begin() + 2*chunkSize);                                    

        //Computes the interesection and give preference to the middle of the image to focus in on the subject
        float distance = 0.4 * computeHistIntersection(targetFull, curFull) 
            + 0.6 * computeHistIntersection(targetMiddle,curMiddle);

        std::pair<float, std::string> curPair(distance, filenames[i]);
        pairs.push_back(curPair);
    }
    std::sort(pairs.begin(), pairs.end());

    return pairs;
}

/*
  Task 4: Texture and Color
*/
std::vector<std::pair<float, std::string>> texturedMatching(cv::Mat targetImg, std::vector<char *> filenames, std::vector<std::vector<float>> data) {
    //Compute target image texture features
    cv::Mat texturedImg;
    textureSobelMagnitude(targetImg, texturedImg);
    std::vector<float> targetTextureVec = computeTexturedHistogram(texturedImg);

    //Compute target image color features
    std::vector<float> targetColorVec = extractHist(targetImg);

    int textureSize = targetTextureVec.size();

    std::vector<std::pair<float, std::string>> pairs;

    for(int i = 0; i < filenames.size(); i++) {

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

    // Sorts vector of distances
    std::sort(pairs.begin(), pairs.end());

    return pairs;
}

/*
  Task 5: Deep Network embedding
*/
std::vector<std::pair<float, std::string>> dnnMatching(std::string imgArg, std::vector<char *> filenames, std::vector<std::vector<float>> data) {
    std::vector<float> targetV;

    // Finds the target image vector in the csv file
    for (int i = 0; i < filenames.size(); i++) {
        if (strcmp(filenames[i], imgArg.c_str()) == 0) {
            targetV = data[i];

            printf("\nTarget %s Found\n\n", imgArg.c_str());
            break;
        } 
    }

    // Mormalizes the target vector
    normalizeV(targetV, targetV);

    // Declare a new vector of pairs containing the distance to target and filename.
    std::vector<std::pair<float, std::string>> pairs;

    // Declares a vector to hold the current normalized vector from the loop.
    std::vector<float> currNormV;

    for (int i = 0; i < filenames.size(); i++) {

        // Checks that img vector being compared is not the target img vector.
        if (strcmp(filenames[i], imgArg.c_str()) != 0) {

            // Fetches and normalizes vector img being compared to target img
            std::vector<float> currV = data[i];
            normalizeV(currV, currNormV);

            // Computes the dot product of the normalized target img vector and img vector being compared
            float sum = 0;
            for (int j = 0; j < targetV.size(); j++) {
                sum += targetV[j] * currNormV[j];
            }

            // Adds cosign difference and file name to a pair
            std::pair<float, std::string> currPair((1 - sum), filenames[i]);

            // Adds the above pair to a vector
            pairs.push_back(currPair);
        }
    }
    // Sorts the vector of pairs.
    std::sort(pairs.begin(), pairs.end());
    
    return pairs;
}

/*
  Prints the N file names with the smallest distance to target.

  int N the number of filenames to print.
*/
void printFilenames(int N, std::vector<std::pair<float, std::string>> pairs) {
    for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }
}

/*
  A helper function for task 5 that normazlizes a given vector.

  std::vector<float> srcVector the vector that needs to be normalized.
  std::vector<float> dstVector vector that holds the normalized vector.
  returns int value indication of success or failure of running normalization.
*/
static int normalizeV(std::vector<float> srcVector, std::vector<float> &dstVector) {
    if (srcVector.empty()) {
        printf("Source vector contains no values\n");
        return -1;
    }

    // Normalizes the vectors
    // Calculuate the sum of squares
    float sum = 0;

    for (int i = 0; i < srcVector.size(); i++) {
        sum += srcVector[i] * srcVector[i];
    }

    // Calculate length of vector 
    float length = sqrt(sum);

    for (int i = 0; i < srcVector.size(); i++) {
        srcVector[i] = srcVector[i] / length;
    }

    dstVector = srcVector;

    return 0;
}

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

    // Task 1: Baseline Matching
    if(method == "b"){
        // Computes distances and returns a pair of distances and filenames.
        std::vector<std::pair<float, std::string>> pairs = baselineMatching(targetImg, filenames, data);

        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    }

    // Task 2: Histogram Matching
    else if(method == "h"){
        // Computes distances and returns a pair of distances and filenames.
        std::vector<std::pair<float, std::string>> pairs = histMatching(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    }

    // Task 2: 3x3 histogram matching.
    else if (method == "h3"){
        // Computes distances and returns a pair of distances and filenames.
        std::vector<std::pair<float, std::string>> pairs = hist3x3Matching(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);

    }

    //Task 3: Multiple Histogram Matching
    else if (method == "mh") {
        // Computes distances and returns a pair of distances and filenames.
        std::vector<std::pair<float, std::string>> pairs = multiHistMatching(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    }
     
    //Task 3: Multiple Histogram Matching - pass 'mh2' as the method
    else if (method == "mh2"){
        // Computes distances and returns a pair of distances and filenames.
        std::vector<std::pair<float, std::string>> pairs = multiHistFullMiddleMatch(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    }

    //Task 4: Texture and Color
    else if (method == "tx") {
        // Computes distances and returns a pair of distances and filenames.
        std::vector<std::pair<float, std::string>> pairs = texturedMatching(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    } 

    // Task 5: Deep Network embedding
    else if (method == "rc") {
        // Computes distances and returns a pair of distances and filenames.
        std::vector<std::pair<float, std::string>> pairs = dnnMatching(imgArg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);

       
    } 
    
    // Task 6: Compare DNN Embedding and Classic Features
    else if (method == "dvf") {


    }
    return 0;
}