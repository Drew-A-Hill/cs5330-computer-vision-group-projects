/*
  Drew Hill & Abhiram Banda,
  Summer 2026
  CS 5330 Computer Vision

  src/ar.cpp

  Holds AR logic. Loads saved intrinsics from calibration. Computes the targets pose for 
  each frame, then projects and draes 3D object onto the board.
*/

#include <opencv2/opencv.hpp>
#include "calibration.h"
#include <vector>
using namespace cv;
using namespace std;


/*
  Reads the saved camera intrinsics file. If the file has no orientation entry sets 
  a default orientation of 0.

  std::string path - path to the intrinsics file.
  char &orientation - orientation used during calibration.
  returns intrinsics read from file as well as orientation if found returns empty pair on
  read failure.
*/
pair<Mat, vector<double>> read_intrinsics(string path, char &orientation) {
    pair<Mat, vector<double>> intrinsics;

     FileStorage read(path, FileStorage::READ);
     if (!read.isOpened()) {
        cout << "Can Not Find File Path: " << path << endl;
        return intrinsics;
     }

    Mat camera_matrix;
    vector<double> distortion;
    string ori;

    // Pull each value out by the same key names used when writing.
    read["camera_matrix"] >>  camera_matrix;
    read["distortion"] >> distortion;
    read["orientation"] >> ori;

    intrinsics.first = camera_matrix;
    intrinsics.second = distortion;

    if (ori.empty()) {
        orientation = '0';

    } else {
        orientation = ori[0];
    }

    return intrinsics;
}

/*
  Detects the target and if found computes its pose relative to the camera for the 
  current frame.

  cv::Mat &frame - current frame.
  cv::Size size - size used during calibration.
  std::vector<Vec3f> point_set - 3D points of the corners.
  std::vector<Point2f> &corners - detected corners.
  std::pair intrinsics - intrinsics camera_matrix and distortion.
  cv::Mat &rvec - rotation of chessboard relatice to camera.
  cv::Mat &tvec -  location of chessboard relatice to camera.
  returns true if a pose was found this frame
*/
bool compute_pose(
    Mat &frame, 
    Size size, 
    vector<Vec3f> point_set, 
    vector<Point2f> &corners, 
    pair<Mat, vector<double>> intrinsics,
    Mat &rvec,
    Mat &tvec
    ) {

    bool detected = detect_corners(frame, size, corners);
    bool solved;

        if (detected) {
            solved = solvePnP(
                point_set, 
                corners, 
                intrinsics.first, 
                intrinsics.second, 
                rvec, 
                tvec
            );
        } else {
            return false;
        }

    return  solved;
}

/*
  Projects the 3D coordinate axes at the target origin onto the image and draws them as 
  colored arrows.

  cv::Mat &frame - current frame.
  cv::Mat &rvec - rotation of chessboard relatice to camera.
  cv::Mat &tvec -  location of chessboard relatice to camera.
  std::pair intrinsics - intrinsics camera_matrix and distortion.
*/
void draw_axis(Mat &src, Mat &rvec, Mat &tvec, pair<Mat, vector<double>> &intrinsics) {
    vector<Point2f> img_points;
    vector<Vec3f> points = {
        Vec3f(0, 0, 0),
        Vec3f(-2, 0, 0),
        Vec3f(0, -2, 0),
        Vec3f(0, 0, -2)
    }; 

    projectPoints(points, rvec, tvec, intrinsics.first, intrinsics.second, img_points);

    // Draws axes
    arrowedLine(src, img_points[0], img_points[1], Scalar(0, 255, 0), 2);
    arrowedLine(src, img_points[0], img_points[2], Scalar(255, 0, 0), 2);
    arrowedLine(src, img_points[0], img_points[3], Scalar(0, 0, 255), 2);
}