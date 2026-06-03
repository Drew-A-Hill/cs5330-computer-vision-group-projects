#include <opencv2/opencv.hpp>
#include <iostream>
#include "features.h"
#include <cstdio>

/*
    Program which holds the logic for extracting vectors from an image
*/

//Takes an image and returns a vector containing the color channels of the 49 pixels in the middle of the image
std::vector<float> extractBaselineFeature(cv::Mat &image){
    //calculating the center of the image
    int centerRow = image.rows / 2;
    int centerCol = image.cols / 2;
    std::vector<float> res;

    //loops through the 49 pixels in the center of the image 7x7
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

//Computes the sum of squared distances between two vectors
float ssd(std::vector<float> &vec1, std::vector<float> &vec2){
    float res = 0;
    for(int i = 0; i < vec1.size(); i++){
        float diff = vec1[i] - vec2[i];
        res += (diff * diff);
    }
    return res;
}

/*
    Extracts the color histogram of an image which we can then use to compare with another histogram. 
    We return a vector which contains the values of the histogram
*/
std::vector<float> extractHist(cv::Mat &image){
    //Initializes a 2D Mat of size 16x16 intialized to all 0's
    cv::Mat hist = cv::Mat::zeros(16, 16, CV_32F);

    //Loops through every pixel in the image
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            cv::Vec3b* row = image.ptr<cv::Vec3b>(i);
            uchar B = row[j][0];
            uchar G = row[j][1];
            uchar R = row[j][2];
            int total = R + G + B;

            //Skips black pixels
            if(total == 0){
                continue;
            }

            //Computes the percentage of red and green values in the pixel
            float r = (float)R / total;
            float g = (float)G / total;

            //Gets the bins of the red and green values
            int binR = (int)(r * 16);
            if(binR == 16){
                binR = 15;
            }
            int binG = (int)(g * 16);
            if(binG == 16){
                binG = 15;
            }

            //Increments the histogram value at binR and binG
            hist.at<float>(binR,binG)++;
        }
    }
    int totalPixels = image.rows * image.cols;
    //Normalizes the values of the hist
    hist/=totalPixels;

    //Pushes the values of the hist into a vector which we'll return
    std::vector<float> res;                 
    for (int i = 0; i < 16; i++) {                                                                        
        for (int j = 0; j < 16; j++) {                                                                    
            res.push_back(hist.at<float>(i, j));                                                          
        }                                                                                                 
    }                                                                                                     
    return res;       

}

/*
    Similar to the previous function except this one creates a 3d Mat of size 8x8x8. And instead of jsut r and g values,
    we compute the whole image RGB histogram of an image. 
*/
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
    //Normalize the hist
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

//Computes the distance between two given vectors representing the histograms of images. 
float computeHistIntersection(std::vector<float> &hist1, std::vector<float> &hist2){
    float res = 0;
    for(int i = 0; i < hist1.size(); i++){
        res += std::min(hist1[i], hist2[i]);
    }
    return 1 - res;
}

/*
    Task 3: Multi Histogram - Takes an image and breaks it into 4 images: top, bottom, left, and right halves and extracts
    the hist of each of those separate images. Then we add the values of each of those vectors into a single vector
*/
std::vector<float> extractMultipleHistQuadrants(cv::Mat &image){
    //Logic for grabbing each of the halves
    cv::Mat topHalf = image(cv::Rect(0, 0, image.cols, image.rows/2));
    cv::Mat bottomHalf = image(cv::Rect(0, image.rows / 2, image.cols, image.rows / 2));
    cv::Mat rightHalf = image(cv::Rect(image.cols/2, 0, image.cols/2, image.rows));
    cv::Mat leftHalf = image(cv::Rect(0, 0, image.cols/2, image.rows));

    //Extracts the hist/vector of each of the broken images
    std::vector<float> topVector = extractHist(topHalf);
    std::vector<float> bottomVector = extractHist(bottomHalf);
    std::vector<float> leftVector = extractHist(leftHalf);
    std::vector<float> rightVector = extractHist(rightHalf);

    //Adds the values of the broken vectors into one
    std::vector<float> vec;
    vec.insert(vec.end(), topVector.begin(), topVector.end());               
    vec.insert(vec.end(), bottomVector.begin(), bottomVector.end());         
    vec.insert(vec.end(), leftVector.begin(), leftVector.end());             
    vec.insert(vec.end(), rightVector.begin(), rightVector.end());
    
    return vec;      
}

/*
    Task 3. Very similar to the previous function except we break the image into the full image and the center of the image
    rather than breaking into 4 halves. After testing, we noticed that this extraction gives much better results
*/
std::vector<float> extractMultipleHistFullMiddle(cv::Mat &image){
    std::vector<float> fullImgVec = extractHist(image);
    int cropWidth = image.cols / 2;
    int cropHeight = image.rows / 2;
    //Calculate the top-left starting coordinates (X, Y)
    int startX = (image.cols - cropWidth) / 2;
    int startY = (image.rows - cropHeight) / 2;
    //Extract the middle region
    cv::Mat middleRegion = image(cv::Rect(startX, startY, cropWidth, cropHeight));
    std::vector<float> middleImgVec = extractHist(middleRegion);

    std::vector<float> vec;
    vec.insert(vec.end(), fullImgVec.begin(), fullImgVec.end());               
    vec.insert(vec.end(), middleImgVec.begin(), middleImgVec.end());         

    return vec;      
}


/*
  Task 4: Creates a Sobel Gradient Magnitude textured image from a source image.
  Applies Sobel X and Y filters, computes the magnitude, and normalizes the result.
*/
int textureSobelMagnitude(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        printf("Frame is empty\n");
        return -1;
    }

    cv::Mat gray, sx, sy, mag;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Compute Sobel X and Y using OpenCV
    cv::Sobel(gray, sx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sy, CV_32F, 0, 1, 3);

    // Compute gradient magnitude
    cv::magnitude(sx, sy, mag);

    // Normalize the magnitude image
    cv::normalize(mag, dst, 0, 255, cv::NORM_MINMAX, CV_8U);

    return 0;
}

/*
  Computes a histogram for textured images.

  cv::Mat &src Source image to be converted.
  returns std::vector the calculated histogram vector
*/
std::vector<float> computeTexturedHistogram(cv::Mat &src) {
    // Declares histogram matrix
    cv::Mat hist;

    // Sets parameters used for converting textured image into histogram
    int channels[] = {0};
    int bins = 32;
    int size[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range};

    // Converts textured image into histogram.
    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, size, ranges);

    // Declares vector to hold feature values
    std::vector<float> textured;

    // Adds feature values from histogram to vector
    for (int i = 0; i < hist.rows; i++) {
        float val = hist.at<float>(i, 0);

        textured.push_back(val);
    }

    return textured;
}
