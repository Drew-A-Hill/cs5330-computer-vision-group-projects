/*
  Drew Hill & Abhiram Banda,
  Summer 2026
  CS 5330 Computer Vision

  src/calibration.cpp

  Holds calibration logic. Performs corner detection, building point set, storing views, 
  and running the camera calibration to produce the intrinsic parameters.
*/

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;


/*
  Detects the chessboards corners in on a grayscale frame. Indicates true if corners are
  detected in the frame and false if not.

  Mat &src - color frame.
  Size size - size of the corners.
  vector<Point2f> &corners - detected corners.
  returns true only if the full chessboard was found else returns false.
*/
bool detect_corners(Mat &src, Size size, vector<Point2f> &corners) {
    if (src.empty()) {
        printf("No frame found");
        return false;
    }

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    bool detected = findChessboardCorners(
        gray, 
        size, 
        corners, 
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK
    );

    if(detected) {
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));
        return detected;
    }
    
    //Returns false if corners are not detected
    return false;
}

/*
  Builds the 3D coordinates of the targets corners. 

  Size size - internal-corner grid (9x6).
  returns one Vec3f per corner.
*/
vector<Vec3f> point_set(Size size) {
    vector<Vec3f> points;

    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            points.push_back(Vec3f(i, -j, 0));
        }
    }

    return points;
}

/*
  Adds the point set vector to a vector containing all point sets that have been detected.

  std::vector<vector<Vec3f>> &point_list - the current vector of points prior to adding new point set.
  std::vector<Vec3f> &point_set - point set to be added if detected.
  bool detected - indicates if corners detected.
  returns a vector of point sets.
*/
vector<vector<Vec3f>> point_list(vector<vector<Vec3f>> &point_list, vector<Vec3f> &point_set, bool detected) {
    if (detected) {
        point_list.push_back(point_set);
    }

    return point_list;
}

/*
  Adds the vector of corners to a vector containing all corners that have been detected.

  std::vector<vector<Point2f>> corner_list - the current vector of corners prior to adding new corners.
  std::vector<Point2f> corners - corners to be added if detected.
  bool detected - indicates if corners detected.
  returns a vector of corners.
*/
vector<vector<Point2f>> corner_list(vector<vector<Point2f>> corner_list, vector<Point2f> corners, bool detected) {
    if (detected) {
        corner_list.push_back(corners);
    }

    return corner_list;
}

/*
  Runs the camera calibration over all saved views and returns the intrinsics.

  Mat &src - a frame.  
  vector<vector<Point2f>> corner_list - 2D corners per saved view.
  vector<vector<Vec3f>> point_list - matching 3D points per saved view.
  returns intrinsics camera_matrix and  distortion.
*/
pair<Mat, vector<double>> calibrate_camera(Mat &src, vector<vector<Point2f>> corner_list, vector<vector<Vec3f>> point_list) {
    // Initializes a camera matrix
    Mat camera_matrix(Size(3, 3), CV_64FC1);

    // Creates the camera matrix
    camera_matrix.ptr<double>(0)[0] = 1;
    camera_matrix.ptr<double>(0)[1] = 0;
    camera_matrix.ptr<double>(0)[2] = src.cols / 2;

    camera_matrix.ptr<double>(1)[0] = 0;
    camera_matrix.ptr<double>(1)[1] = 1;
    camera_matrix.ptr<double>(1)[2] = src.rows / 2;

    camera_matrix.ptr<double>(2)[0] = 0;
    camera_matrix.ptr<double>(2)[1] = 0;
    camera_matrix.ptr<double>(2)[2] = 1;

    // Initializes distortion vector.
    vector<double> distortion(0);

    cout << "-------- Before Calibration --------" << endl;
    cout << "Camera Matrix: \n" << camera_matrix << endl;
    cout << "\nDistortion: \n" << Mat(distortion) << endl;

    vector<Mat> rvecs;
    vector<Mat> tvecs;
    double error = calibrateCamera(point_list, corner_list, src.size(), camera_matrix, distortion, rvecs, tvecs);

    cout << "\n-------- After Calibration --------" << endl;
    cout << "Camera Matrix: \n" << camera_matrix << endl;
    cout << "\nDistortion: \n" << Mat(distortion) << endl;
    cout << "\nError: " << error << endl;

    // Adds intrinsics to pair to be returned
    pair<Mat, vector<double>> details(camera_matrix, distortion);
    return details;
}