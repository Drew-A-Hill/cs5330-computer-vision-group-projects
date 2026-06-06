/*
  Abhiram Banda, Drew Hill
  Summer 2026
  CS 5330 Computer Vision

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
#include <set>

namespace fs = std::filesystem;

/*
  Task 1: Baseline Matching
  This method computes the baseline feature diffence between the target image and each of the other
  images in the data base.

  cv::Mat targetImg a Mat containging the target image.
  std::vector<char *> all filenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> data a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename.
*/
std::vector<std::pair<float, std::string>> baselineMatching(cv::Mat targetImg, std::vector<char *> filenames, std::vector<std::vector<float>> data) {

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

  cv::Mat targetImg a Mat containging the target image.
  std::vector<char *> all filenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> data a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename.
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

  cv::Mat targetImg a Mat containging the target image.
  std::vector<char *> all filenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> data a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename.
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

  cv::Mat targetImg a Mat containging the target image.
  std::vector<char *> all filenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> data a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename.
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

  cv::Mat targetImg a Mat containging the target image.
  std::vector<char *> all filenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> data a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename.
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

  cv::Mat targetImg a Mat containging the target image.
  std::vector<char *> all filenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> data a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename.
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
  Task 5: Deep Network embedding
  Function is used to compute the similar images vector using a dnn embedding. Uses cosign difference to compute
  the difference between an image and the target image.

  std::string imgArg the file name for the target image.
  std::vector<char *> all filenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> data a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename.
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

    // Normalizes the target vector
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
  Task 7: Custom Flower Matching
  Combines center-focused color histogram, texture histogram, and DNN embeddings.
  Flowers have distinctive bright colors (especially in the center of the image),
  petal texture patterns, and green foliage backgrounds.

  cv::Mat targetImg a Mat containging the target image.
  std::string imgArg the file name for the target image.
  std::vector<char *> all dnnFilenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> dnnData a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename.
*/
std::vector<std::pair<float, std::string>> customMatching(
    cv::Mat targetImg, std::string imgName,
    std::vector<char *> dnnFilenames, std::vector<std::vector<float>> dnnData) {

    // Compute target center-focused color histogram (full + middle)
    std::vector<float> targetColorVec = extractMultipleHistFullMiddle(targetImg);
    int chunkSize = targetColorVec.size() / 2;
    std::vector<float> targetFull(targetColorVec.begin(), targetColorVec.begin() + chunkSize);
    std::vector<float> targetMiddle(targetColorVec.begin() + chunkSize, targetColorVec.end());

    // Find and normalize target DNN embedding
    std::vector<float> targetDnnNorm;
    for (int i = 0; i < dnnFilenames.size(); i++) {
        if (strcmp(dnnFilenames[i], imgName.c_str()) == 0) {
            normalizeV(dnnData[i], targetDnnNorm);
            break;
        }
    }

    // Build a map from filename -> index for DNN data
    std::unordered_map<std::string, int> dnnMap;
    for (int i = 0; i < dnnFilenames.size(); i++) {
        dnnMap[dnnFilenames[i]] = i;
    }

    std::vector<std::pair<float, std::string>> pairs;

    for (int i = 0; i < dnnFilenames.size(); i++) {
        std::string fname = dnnFilenames[i];
        if (fname == imgName) continue;

        // DNN cosine distance
        std::vector<float> curDnnNorm;
        normalizeV(dnnData[i], curDnnNorm);
        float dot = 0;
        for (int j = 0; j < targetDnnNorm.size(); j++) {
            dot += targetDnnNorm[j] * curDnnNorm[j];
        }
        float dnnDistance = 1 - dot;

        // Center-focused color distance (flower is the centered subject)
        cv::Mat img = cv::imread("olympus/" + fname);
        float colorDistance = 1.0;
        if (!img.empty()) {
            std::vector<float> curColorVec = extractMultipleHistFullMiddle(img);
            std::vector<float> curFull(curColorVec.begin(), curColorVec.begin() + chunkSize);
            std::vector<float> curMiddle(curColorVec.begin() + chunkSize, curColorVec.end());
            colorDistance = 0.3 * computeHistIntersection(targetFull, curFull)
                          + 0.7 * computeHistIntersection(targetMiddle, curMiddle);
        }

        // DNN for semantics (knows what flowers are), color for visual similarity
        float combined = 0.6 * dnnDistance + 0.4 * colorDistance;

        pairs.push_back({combined, fname});
    }

    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

/*
  Extension:
  Function is used to compute the similar images vector using a dnn embedding. Uses sum of squared 
  differences difference to compute the difference between an image and the target image.

  std::string imgArg the file name for the target image.
  std::vector<char *> all filenames in the data bases being used for comparisons.
  std::vector<std::vector<float>> data a vector that contains each vector of data for each image.
  returns std::vector<std::pair<float, std::string>> a vector of pairs containing the distance to target and filename. 
*/
std::vector<std::pair<float, std::string>> dnnSSDMatching(std::string imgArg, std::vector<char *> filenames, std::vector<std::vector<float>> data) {
    std::vector<float> targetV;

    // Finds the target image vector in the csv file
    for (int i = 0; i < filenames.size(); i++) {
        if (strcmp(filenames[i], imgArg.c_str()) == 0) {
            targetV = data[i];

            printf("\nTarget %s Found\n\n", imgArg.c_str());
            break;
        } 
    }

    // Declare a new vector of pairs containing the distance to target and filename.
    std::vector<std::pair<float, std::string>> pairs;

    for (int i = 0; i < filenames.size(); i++) {

        // Checks that img vector being compared is not the target img vector.
        if (strcmp(filenames[i], imgArg.c_str()) != 0) {
            // Computes the sum of squared differnces
            float sumSD = ssd(targetV, data[i]);

            // Adds sdd value and filename to the pair
            std::pair<float, std::string> currPair(sumSD, filenames[i]);

            // Adds the above pair to a vector
            pairs.push_back(currPair);
        }
    }
    // Sorts the vector of pairs.
    std::sort(pairs.begin(), pairs.end());

    return pairs;
}

/*
  Extension2
  This function builds a GUI that displays the target image, the N most similar images, 
  as well as each of their distance values.

  std::string filepath the file path of the target image to be displayed.
  std::vector<std::pair<float, std::string>> pairs a vector containing paris with distance and filename.
  int N the number of images in addition to the target image to display.
*/
void gui(std::string filepath, std::vector<std::pair<float, std::string>> pairs, int N) {
    // Reads target image.
    cv::Mat targetImg = cv::imread(filepath);

    // Displays target image.
    cv::imshow("Target - " + filepath, targetImg);

    for (int i = 0; i < N; i++) {
        // Reads the Nth most similar image file.
        cv::Mat image = cv::imread("olympus/" + pairs[i].second);

        // Builds window title from file name and distance.
        std::string title = pairs[i].second + " (" + std::to_string(pairs[i].first) + ")";

        // Displays the Nth mosth similar image.
        cv::imshow(title, image);
    }
    cv::waitKey(0);
}

/*
  Helper function that prints the N file names with the smallest distance to target.

  int N the number of filenames to print.
*/
void printFilenames(int N, std::vector<std::pair<float, std::string>> pairs) {
    for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }
}

/*
  Main function that takes arguments to control image comparisons. 
  run ./matcher olympus/{img name.jpg} features.csv {N top imgs to display} action (ie tx for texture)
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

    // Declares a vector of pairs containing distance and filename.
    std::vector<std::pair<float, std::string>> pairs;

    // Task 1: Baseline Matching
    if(method == "b"){
        // Computes distances and returns a pair of distances and filenames.
        pairs = baselineMatching(targetImg, filenames, data);

        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    }

    // Task 2: Histogram Matching
    else if(method == "h"){
        // Computes distances and returns a pair of distances and filenames.
        pairs = histMatching(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    }

    // Task 2: 3x3 histogram matching.
    else if (method == "h3"){
        // Computes distances and returns a pair of distances and filenames.
        pairs = hist3x3Matching(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);

    }

    //Task 3: Multiple Histogram Matching
    else if (method == "mh") {
        // Computes distances and returns a pair of distances and filenames.
        pairs = multiHistMatching(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    }
     
    //Task 3: Multiple Histogram Matching - pass 'mh2' as the method
    else if (method == "mh2"){
        // Computes distances and returns a pair of distances and filenames.
        pairs = multiHistFullMiddleMatch(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    }

    //Task 4: Texture and Color
    else if (method == "tx") {
        // Computes distances and returns a pair of distances and filenames.
        pairs = texturedMatching(targetImg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    } 

    // Task 5: Deep Network embedding
    else if (method == "rc") {
        // Computes distances and returns a pair of distances and filenames.
        pairs = dnnMatching(imgArg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    } 
    
    // Task 6: Compare DNN Embedding and Classic Features
    else if (method == "dvf") {
        // Declares vectors that hold all of the data from each features csv file.
        std::vector<std::vector<float>> dnnData; 
        std::vector<std::vector<float>> baselineData;
        std::vector<std::vector<float>> histData;
        std::vector<std::vector<float>> hist3x3Data;
        std::vector<std::vector<float>> multiHistData;
        std::vector<std::vector<float>> textureData;

        // Declares vectors that hold the file names for each features csv file.
        std::vector<char *> dnnFilenames;
        std::vector<char *> baselineFilenames;
        std::vector<char *> histFilenames;
        std::vector<char *> hist3x3Filenames;
        std::vector<char *> multiHistFilenames;
        std::vector<char *> textureFilenames;

        // Reads the csv and assigns the data and filenames to the correct variable declared above.
        read_image_data_csv( (char*)"ResNet18_olym.csv", dnnFilenames, dnnData);
        read_image_data_csv((char*)"features.csv", baselineFilenames, baselineData);
        read_image_data_csv((char*)"histogram_features.csv", histFilenames, histData);
        read_image_data_csv((char*)"histogram_features_3.csv", hist3x3Filenames, hist3x3Data);
        read_image_data_csv((char*)"mult_hist_1.csv", multiHistFilenames, multiHistData);
        read_image_data_csv((char*)"texture.csv", textureFilenames, textureData);
        
        // Declares variables to hold compared vectors of pairs.
        std::vector<std::pair<float, std::string>> dnnPairs;
        std::vector<std::pair<float, std::string>> baselinePairs;
        std::vector<std::pair<float, std::string>> colorHistpairs;
        std::vector<std::pair<float, std::string>> hist3x3Pairs;
        std::vector<std::pair<float, std::string>> multiHistPairs;
        std::vector<std::pair<float, std::string>> texturePairs;

        // Holds target filename in a string format.
        std::string imgBase = fs::path(imgArg).filename().string();

        // Compute the similar images vector using a dnn embedding.
        dnnPairs = dnnMatching(imgBase, dnnFilenames, dnnData);

        // Reads the target image into a Mat.
        cv::Mat target = cv::imread("olympus/" + imgBase);

        // Check to ensure the image has been loaded.
        if (target.empty()) {
            printf("Missing Target\n");
            return-1;
        }

        // Performs comparisons using the different feature types and stores then in a variable.
        baselinePairs = baselineMatching(target, baselineFilenames, baselineData);
        colorHistpairs = histMatching(target, histFilenames, histData);
        hist3x3Pairs = hist3x3Matching(target, hist3x3Filenames, hist3x3Data);
        multiHistPairs = multiHistMatching(target, multiHistFilenames, multiHistData);
        texturePairs = texturedMatching(target, textureFilenames, textureData);

        // Prints the filename of the target image being used.
        printf("Target Image: %s\n", imgArg.c_str());

        // Retrieves and prints the top N closest matches to the target image.
        for (int j = 0; j < N; j++) {
            std::string dnnOut = dnnPairs[j].second;
            std::string baselineOut = baselinePairs[j + 1].second;
            std::string histOut = colorHistpairs[j + 1].second;
            std::string hist3x3Out = hist3x3Pairs[j + 1].second;
            std::string multiOut = multiHistPairs[j + 1].second;
            std::string texturedOut = texturePairs[j + 1].second;


            printf("DNN:%s\tBaseline:%s\tHist:%s\tHist3x3:%s\tMulti:%s\tTexture:%s\n\n", 
                    dnnOut.c_str(), baselineOut.c_str(), histOut.c_str(), 
                    hist3x3Out.c_str(), multiOut.c_str(), texturedOut.c_str());
        }
    }

    // Task 7: Custom License Plate Matching (texture + center color + DNN)
    // Usage: ./matcher olympus/pic.0482.jpg ResNet18_olym.csv 5 lp
    else if (method == "lp") {
        // Extract just the filename (e.g. "pic.0482.jpg") for DNN lookup
        std::string imgName = fs::path(imgArg).filename().string();

        pairs = customMatching(targetImg, imgName, filenames, data);

        printf("Top %d most similar:\n", N);
        printFilenames(N, pairs);

        printf("\nTop %d least similar:\n", N);
        for (int i = pairs.size() - 1; i >= (int)pairs.size() - N; i--) {
            printf("%s\n", pairs[i].second.c_str());
        }
    } 

    // Extension1 - Using sum of squared differences on dnn embedding to find distances.
    else if (method == "res2") {
        // Computes distances and returns a pair of distances and filenames.
        pairs = dnnSSDMatching(imgArg, filenames, data);
        
        // Prints the N file names with the smallest distance to target.
        printFilenames(N, pairs);
    } 

    // Extension 2 - GUI that displays the target image N most similar images.
    // Displays the images.
    gui(imgArg, pairs, N);

    return 0;
}