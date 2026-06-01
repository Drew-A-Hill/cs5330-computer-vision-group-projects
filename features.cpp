#include <opencv2/opencv.hpp>
#include <iostream>
#include "features.h"
#include <cstdio>

std::vector<float> extractBaselineFeature(cv::Mat &image){
    int centerRow = image.rows / 2;
    int centerCol = image.cols / 2;
    std::vector<float> res;

    for(int i = centerRow-3; i <= centerRow+3; i++){
        for(int j = centerCol-3; j <= centerCol+3; j++){
            cv::Vec3b* row = image.ptr<cv::Vec3b>(i);
            for (int c = 0; c<image.channels(); c++){
                res.push_back(float(row[j][c]));
            }
        }
    }
    return res;
}

float ssd(std::vector<float> &vec1, std::vector<float> &vec2){
    float res = 0;
    for(int i = 0; i < vec1.size(); i++){
        float diff = vec1[i] - vec2[i];
        res += (diff * diff);
    }
    return res;
}

std::vector<float> extractHist(cv::Mat &image){
    cv::Mat hist = cv::Mat::zeros(16, 16, CV_32F);
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            cv::Vec3b* row = image.ptr<cv::Vec3b>(i);
            uchar B = row[j][0];
            uchar G = row[j][1];
            uchar R = row[j][2];
            int total = R + G + B;
            if(total == 0){
                continue;
            }
            float r = (float)R / total;
            float g = (float)G / total;

            int binR = (int)(r * 16);
            if(binR == 16){
                binR = 15;
            }
            int binG = (int)(g * 16);
            if(binG == 16){
                binG = 15;
            }

            hist.at<float>(binR,binG)++;
        }
    }
    int totalPixels = image.rows * image.cols;
    hist/=totalPixels;

    std::vector<float> res;                 
    for (int i = 0; i < 16; i++) {                                                                        
        for (int j = 0; j < 16; j++) {                                                                    
            res.push_back(hist.at<float>(i, j));                                                          
        }                                                                                                 
    }                                                                                                     
    return res;       

}

std::vector<float> extractHist3x3(cv::Mat &image){
    int sizes[] = {8, 8, 8};
    // Initialize a 3D Mat with dimensions (3), the size array, data type (float), and fill with 0s
    cv::Mat hist(3, sizes, CV_32FC1, cv::Scalar(0));
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            
            cv::Vec3b* row = image.ptr<cv::Vec3b>(i);
            uchar B = row[j][0];
            uchar G = row[j][1];
            uchar R = row[j][2];
            int total = R + G + B;

            int binR = (int)(R / 32);
            if(binR == 8){
                binR = 7;
            }
            int binG = (int)(G / 32);
            if(binG == 8){
                binG = 7;
            }
            int binB = (int)(B / 32);
            if(binB == 8){
                binB = 7;
            }

            hist.at<float>(binR,binG,binB)++;
        }
    }
    int totalPixels = image.rows * image.cols;
    hist/=totalPixels;

    std::vector<float> res;                 
    for (int i = 0; i < 8; i++) {                                                                        
        for (int j = 0; j < 8; j++) { 
            for(int k = 0; k < 8; k++){
                res.push_back(hist.at<float>(i, j, k));
            }                                                                                                               
        }                                                                                                 
    }                                                                                                     
    return res;       
}

float computeHistIntersection(std::vector<float> &hist1, std::vector<float> &hist2){
    float res = 0;
    for(int i = 0; i < hist1.size(); i++){
        res += std::min(hist1[i], hist2[i]);
    }
    return 1 - res;
}
