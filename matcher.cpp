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

// Program that holds the logic for matching the target image with each image in the image database and returns the N most similar images.
int main(int argc, char* argv[]){
    //First arg which is the path to the target image
    std::string imgArg = argv[1];
    cv::Mat targetImg = cv::imread(imgArg);

    //Second arg: path to the csv which holds the feature vectors
    std::string csvPath = argv[2];

    //Third arg: the number of images you want returned
    int N = std::atoi(argv[3]);

    //Foruth arg: the method of computing which relates to the tasks in the assignment
    std::string method = argv[4];

    //First task: Baseline Extraction - pass in 'b' as the method
    if(method == "b"){
        //Creates the vector for the target image using the extractBaselineFeature we created in features.cpp
        std::vector<float> targetImgVector = extractBaselineFeature(targetImg);
        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        //Reads the csv data with the baseline vectors for each image and writes them to the filenames and data vectors.
        read_image_data_csv( (char*)csvPath.c_str(), filenames, data);
        for(int i = 0; i < filenames.size(); i++){
            std::vector<float> &curVec = data[i];
            //Calculates the ssd between the targetImage and the current image in the loop then adds it to the pairs vector
            float distance = ssd(targetImgVector, curVec);
            std::pair<float, std::string> curPair(distance, filenames[i]);
            pairs.push_back(curPair);
        }
        //Finally we sort the pairs vector which will hold the most similar images and the corresponding filename so we can print
        std::sort(pairs.begin(), pairs.end());
        for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }
    }
    //Task 2: Histogram matching - pass 'h' as the method
    else if(method == "h"){
        //Stores the histogram vector of the targetImage which we'll use to compare with each image in the database
        std::vector<float> targetImgHist = extractHist(targetImg);
        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;
        
        //Reads the csv data with the baseline vectors for each image and writes them to the filenames and data vectors.
        read_image_data_csv( (char*)csvPath.c_str(), filenames, data);
        //Loops through each image and its vector/hist
        for(int i = 0; i < filenames.size(); i++){
            std::vector<float> &curImgHist = data[i];
            //Computes the intersection between the targetImage and the current images histogram and pushes into the pairs vector
            float distance = computeHistIntersection(targetImgHist, curImgHist);
            std::pair<float, std::string> curPair(distance, filenames[i]);
            pairs.push_back(curPair);
        }
        //Sorts the pairs vector which we'll use to print the N most similar images
        std::sort(pairs.begin(), pairs.end());
        for(int i = 0; i < N; i ++){
            printf("%s\n", pairs[i].second.c_str());
        }
    }
    //Task 2 - Histogram matching but with a 3d histogram. Pass in 'h3' as the method
    else if (method == "h3"){
        //Holds the histogram for the targetImage
        std::vector<float> targetImgHist = extractHist3x3(targetImg);
        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        read_image_data_csv( (char*)csvPath.c_str(), filenames, data);
        //Loops through each image and its vector/hist
        for(int i = 0; i < filenames.size(); i++){
            std::vector<float> &curImgHist = data[i];
            //Computes the intersection beteween the targetImage and the current Image in the loop
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

        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        read_image_data_csv( (char*)csvPath.c_str(), filenames, data);
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

        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        std::vector<std::pair<float, std::string>> pairs;

        read_image_data_csv( (char*)csvPath.c_str(), filenames, data);

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
    return 0;
}