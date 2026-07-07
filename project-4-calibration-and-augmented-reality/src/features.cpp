/*
  Drew Hill & Abhiram Banda,
  Summer 2026
  CS 5330 Computer Vision

  src/features.cpp

  Separate program that detects robust features (Harris corners and ORB)
  on a video stream. Press 'h' for Harris corners, 'o' for ORB features,
  and 'q' to quit. Use +/- to adjust detection thresholds.
*/

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

/*
  Detects Harris corners on a frame and draws them as circles.

  cv::Mat &src - input color frame.
  cv::Mat &dst - output frame with corners drawn.
  int threshold - Harris response threshold (higher = fewer corners).
*/
void detect_harris(Mat &src, Mat &dst, int threshold) {
    Mat gray, harris_response;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    // Compute Harris corner response
    cornerHarris(gray, harris_response, 2, 3, 0.04);

    // Normalize to 0-255 range
    Mat harris_norm;
    normalize(harris_response, harris_norm, 0, 255, NORM_MINMAX);

    dst = src.clone();

    // Draw circles at points above threshold
    int count = 0;
    for (int i = 0; i < harris_norm.rows; i++) {
        for (int j = 0; j < harris_norm.cols; j++) {
            if ((int)harris_norm.at<float>(i, j) > threshold) {
                circle(dst, Point(j, i), 4, Scalar(0, 0, 255), 1);
                count++;
            }
        }
    }

    putText(dst, "Harris | Threshold: " + to_string(threshold) + " | Points: " + to_string(count),
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
}

/*
  Detects ORB features on a frame and draws keypoints.

  cv::Mat &src - input color frame.
  cv::Mat &dst - output frame with keypoints drawn.
  int max_features - maximum number of ORB features to detect.
*/
void detect_orb(Mat &src, Mat &dst, int max_features) {
    Ptr<ORB> orb = ORB::create(max_features);

    vector<KeyPoint> keypoints;
    orb->detect(src, keypoints);

    drawKeypoints(src, keypoints, dst, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);

    putText(dst, "ORB | Max: " + to_string(max_features) + " | Found: " + to_string(keypoints.size()),
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
}

int main(int argc, char *argv[]) {
    VideoCapture cap;
    Mat img;
    string input_type = "web";
    int wait_val = 1;

    if (argc >= 2) {
        input_type = argv[1];
    }

    if (input_type == "web") {
        cap.open(0);
    } else if (input_type == "mobile") {
        if (argc < 3) {
            cout << "Mobile requires IP: ./features mobile <IP>" << endl;
            return 1;
        }
        string url = "http://" + string(argv[2]) + ":8080/stream.mjpg";
        cap.open(url);
    } else if (input_type == "img") {
        if (argc < 3) {
            cout << "Image requires path: ./features img <path>" << endl;
            return 1;
        }
        img = imread(argv[2]);
        if (img.empty()) {
            cout << "Cannot read image: " << argv[2] << endl;
            return 1;
        }
    } else {
        cout << "Usage: ./features {web|mobile|img} [IP or path]" << endl;
        return 1;
    }

    if (input_type != "img" && !cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return 1;
    }

    // 'h' = Harris, 'o' = ORB
    char mode = 'o';
    int harris_threshold = 150;
    int orb_max_features = 500;
    char rotation = '0';

    cout << "Controls:" << endl;
    cout << "  'h' - Harris corners mode" << endl;
    cout << "  'o' - ORB features mode" << endl;
    cout << "  '+' - increase threshold/features" << endl;
    cout << "  '-' - decrease threshold/features" << endl;
    cout << "  '0'-'3' - rotate frame" << endl;
    cout << "  'q' - quit" << endl;

    for (;;) {
        Mat frame;
        if (input_type == "img") {
            frame = img.clone();
        } else {
            cap >> frame;
        }

        if (frame.empty()) {
            cout << "Empty frame" << endl;
            break;
        }

        if (rotation == '1') {
            rotate(frame, frame, ROTATE_90_CLOCKWISE);
        } else if (rotation == '2') {
            rotate(frame, frame, ROTATE_180);
        } else if (rotation == '3') {
            rotate(frame, frame, ROTATE_90_COUNTERCLOCKWISE);
        }

        Mat dst;
        if (mode == 'h') {
            detect_harris(frame, dst, harris_threshold);
        } else {
            detect_orb(frame, dst, orb_max_features);
        }

        imshow("Features", dst);
        char key = waitKey(wait_val);

        if (key == 'q') {
            break;
        } else if (key == 'h') {
            mode = 'h';
        } else if (key == 'o') {
            mode = 'o';
        } else if (key == '+' || key == '=') {
            if (mode == 'h') {
                harris_threshold = min(harris_threshold + 10, 255);
            } else {
                orb_max_features += 100;
            }
        } else if (key == '0' || key == '1' || key == '2' || key == '3') {
            rotation = key;
        } else if (key == '-') {
            if (mode == 'h') {
                harris_threshold = max(harris_threshold - 10, 10);
            } else {
                orb_max_features = max(orb_max_features - 100, 100);
            }
        }
    }

    return 0;
}
