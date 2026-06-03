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
    //Task 3: Multiple Histogram Matching - pass 'mh' as the method
    else if (method == "mh"){
        //holds the vector for the targetImage whcih we'll use to compare with each image
        std::vector<float> targetImgHist = extractMultipleHistQuadrants(targetImg);
        //Compute chunksize because we broke the image into 4 different images
        int chunkSize = targetImgHist.size() / 4;

        //Uses chunkSize to break the big vector into its 4 separate vectors represingting the portions of the image we chose
        std::vector<float> targetTop(targetImgHist.begin(), targetImgHist.begin() + chunkSize);                              
        std::vector<float> targetBottom(targetImgHist.begin() + chunkSize, targetImgHist.begin() + 2*chunkSize);                                    
        std::vector<float> targetLeft(targetImgHist.begin() + 2*chunkSize, targetImgHist.begin() + 3*chunkSize);                                    
        std::vector<float> targetRight(targetImgHist.begin() + 3*chunkSize, targetImgHist.end());     

        std::vector<std::pair<float, std::string>> pairs;

        //loops through each image and its vector
        for(int i = 0; i < filenames.size(); i++){
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
        std::sort(pairs.begin(), pairs.end());
        for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }
    }
    //Task 3: Multiple Histogram Matching - pass 'mh2' as the method
    else if (method == "mh2"){
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