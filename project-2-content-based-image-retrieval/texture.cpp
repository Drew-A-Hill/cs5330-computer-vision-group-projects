/*


*/

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

/*

*/
std::vector<float> textureHistogram(cv::Mat &src) {
    cv::Mat texturedHist;
    int channels[] = {0};

    float hrange[] = {0, 180};
    float srange[] = {0, 256};

    const float* range[] = {hrange, srange};
    
    int hbins = 30;
    int sbins = 32;

    int hsize[] = {hbins, sbins};

    calcHist(src, 1, channels, cv::Mat(), texturedHist, hsize, range);
}

/*

*/
int magnitudeTexture(Mat &src, Mat &dst) {
    if (src.empty()) {
        printf("Frame is empty\n");
        return -1;
    }

    Mat sx;
    Mat sy;
    Mat mag;

    // Calculates sobel x
    Sobel(src, sx, src.type(), 1, 0);

    // Calculates sobel y
    Sobel(src, sy, src.type(), 0, 1);

    // Calculates the magnitude of the sobel x and y images
    magnitude(sx, sy, mag);

    return 0;
}