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

/*
  Hides the checkerboard target by projecting the outer board boundary and
  filling it with the average color sampled from just outside the boundary.

  cv::Mat &src - current frame.
  cv::Mat &rvec - rotation of chessboard relative to camera.
  cv::Mat &tvec - location of chessboard relative to camera.
  cv::Size size - internal corner grid size (9x6).
  std::pair intrinsics - intrinsics camera_matrix and distortion.
*/
void hide_target(Mat &src, Mat &rvec, Mat &tvec, Size size, pair<Mat, vector<double>> &intrinsics) {
    // Matches point_set convention: (row, -col, 0)
    // Outer board is half a square beyond internal corners in each direction.
    float r_max = size.height - 1;  // max row index (5)
    float c_max = size.width - 1;   // max col index (8)
    float margin = 1.0;
    vector<Vec3f> outer_pts = {
        Vec3f(-margin,        margin, 0),
        Vec3f(r_max + margin, margin, 0),
        Vec3f(r_max + margin, -(c_max + margin), 0),
        Vec3f(-margin,        -(c_max + margin), 0)
    };

    vector<Point2f> img_pts;
    projectPoints(outer_pts, rvec, tvec, intrinsics.first, intrinsics.second, img_pts);

    // Sample average color from just outside the board boundary.
    Scalar avg_color(0, 0, 0);
    int samples = 0;
    for (int i = 0; i < 4; i++) {
        Point2f mid = (img_pts[i] + img_pts[(i + 1) % 4]) * 0.5;
        Point2f center = (img_pts[0] + img_pts[1] + img_pts[2] + img_pts[3]) * 0.25;
        Point2f dir = mid - center;
        Point2f sample_pt = mid + dir * 0.15;
        int sx = (int)sample_pt.x;
        int sy = (int)sample_pt.y;
        if (sx >= 0 && sx < src.cols && sy >= 0 && sy < src.rows) {
            Vec3b c = src.at<Vec3b>(sy, sx);
            avg_color[0] += c[0];
            avg_color[1] += c[1];
            avg_color[2] += c[2];
            samples++;
        }
    }
    if (samples > 0) {
        avg_color /= samples;
    }

    // Fill the board region with the sampled color.
    vector<Point> poly;
    for (auto &p : img_pts) {
        poly.push_back(Point((int)p.x, (int)p.y));
    }
    fillConvexPoly(src, poly, avg_color);
}

/*
  Draws a 3D house with a chimney floating above the board.

  cv::Mat &src - current frame.
  cv::Mat &rvec - rotation of chessboard relative to camera.
  cv::Mat &tvec - location of chessboard relative to camera.
  std::pair intrinsics - intrinsics camera_matrix and distortion.
*/
void draw_virtual_object(Mat &src, Mat &rvec, Mat &tvec, pair<Mat, vector<double>> &intrinsics) {
    // 3D points for the house (z negative = above board)
    vector<Vec3f> pts = {
        // Base floor (z = -0.2, just above board)
        Vec3f(2, -1, -0.2),  // 0: front-left
        Vec3f(6, -1, -0.2),  // 1: front-right
        Vec3f(6, -4, -0.2),  // 2: back-right
        Vec3f(2, -4, -0.2),  // 3: back-left

        // Top of walls (z = -3)
        Vec3f(2, -1, -3),    // 4: front-left top
        Vec3f(6, -1, -3),    // 5: front-right top
        Vec3f(6, -4, -3),    // 6: back-right top
        Vec3f(2, -4, -3),    // 7: back-left top

        // Roof ridge (z = -4.5)
        Vec3f(4, -1, -4.5),  // 8: front ridge
        Vec3f(4, -4, -4.5),  // 9: back ridge

        // Chimney (on the right side of the roof)
        Vec3f(5,   -3.5, -3),    // 10: chimney base front-left
        Vec3f(5.5, -3.5, -3),    // 11: chimney base front-right
        Vec3f(5.5, -4,   -3),    // 12: chimney base back-right
        Vec3f(5,   -4,   -3),    // 13: chimney base back-left
        Vec3f(5,   -3.5, -5),    // 14: chimney top front-left
        Vec3f(5.5, -3.5, -5),    // 15: chimney top front-right
        Vec3f(5.5, -4,   -5),    // 16: chimney top back-right
        Vec3f(5,   -4,   -5),    // 17: chimney top back-left

        // Door (on front wall)
        Vec3f(3.5, -1, -0.2),  // 18: door bottom-left
        Vec3f(4.5, -1, -0.2),  // 19: door bottom-right
        Vec3f(4.5, -1, -1.8),  // 20: door top-right
        Vec3f(3.5, -1, -1.8),  // 21: door top-left
    };

    vector<Point2f> img_pts;
    projectPoints(pts, rvec, tvec, intrinsics.first, intrinsics.second, img_pts);

    // Walls (yellow)
    Scalar wall_color(0, 220, 220);
    // Base edges
    line(src, img_pts[0], img_pts[1], wall_color, 2);
    line(src, img_pts[1], img_pts[2], wall_color, 2);
    line(src, img_pts[2], img_pts[3], wall_color, 2);
    line(src, img_pts[3], img_pts[0], wall_color, 2);
    // Top edges
    line(src, img_pts[4], img_pts[5], wall_color, 2);
    line(src, img_pts[5], img_pts[6], wall_color, 2);
    line(src, img_pts[6], img_pts[7], wall_color, 2);
    line(src, img_pts[7], img_pts[4], wall_color, 2);
    // Vertical edges
    line(src, img_pts[0], img_pts[4], wall_color, 2);
    line(src, img_pts[1], img_pts[5], wall_color, 2);
    line(src, img_pts[2], img_pts[6], wall_color, 2);
    line(src, img_pts[3], img_pts[7], wall_color, 2);

    // Roof (red)
    Scalar roof_color(0, 0, 255);
    line(src, img_pts[4], img_pts[8], roof_color, 2);
    line(src, img_pts[5], img_pts[8], roof_color, 2);
    line(src, img_pts[6], img_pts[9], roof_color, 2);
    line(src, img_pts[7], img_pts[9], roof_color, 2);
    line(src, img_pts[8], img_pts[9], roof_color, 2);

    // Chimney (gray)
    Scalar chimney_color(120, 120, 120);
    // Top
    line(src, img_pts[14], img_pts[15], chimney_color, 2);
    line(src, img_pts[15], img_pts[16], chimney_color, 2);
    line(src, img_pts[16], img_pts[17], chimney_color, 2);
    line(src, img_pts[17], img_pts[14], chimney_color, 2);
    // Verticals
    line(src, img_pts[10], img_pts[14], chimney_color, 2);
    line(src, img_pts[11], img_pts[15], chimney_color, 2);
    line(src, img_pts[12], img_pts[16], chimney_color, 2);
    line(src, img_pts[13], img_pts[17], chimney_color, 2);

    // Door (cyan)
    Scalar door_color(255, 200, 0);
    line(src, img_pts[18], img_pts[19], door_color, 2);
    line(src, img_pts[19], img_pts[20], door_color, 2);
    line(src, img_pts[20], img_pts[21], door_color, 2);
    line(src, img_pts[21], img_pts[18], door_color, 2);
}

/*
  Draws a 3D rocket ship floating above the board with an octagonal body,
  nose cone, fins, a window, and exhaust flames.

  cv::Mat &src - current frame.
  cv::Mat &rvec - rotation of chessboard relative to camera.
  cv::Mat &tvec - location of chessboard relative to camera.
  std::pair intrinsics - intrinsics camera_matrix and distortion.
*/
void draw_rocket(Mat &src, Mat &rvec, Mat &tvec, pair<Mat, vector<double>> &intrinsics) {
    // Center the rocket at (3, -3) on the board
    float cx = 3.0, cy = -3.0;
    float r = 1.0; // radius of octagonal body

    // Build octagon vertices for bottom ring (z = -0.5) and top ring (z = -5)
    // and nose cone tip (z = -6.5)
    vector<Vec3f> pts;

    // 0-7: bottom octagon (z = -0.5)
    // 8-15: top octagon (z = -5)
    for (int ring = 0; ring < 2; ring++) {
        float z = (ring == 0) ? -0.5 : -5.0;
        for (int i = 0; i < 8; i++) {
            float angle = i * CV_PI / 4.0;
            float x = cx + r * cos(angle);
            float y = cy + r * sin(angle);
            pts.push_back(Vec3f(x, y, z));
        }
    }

    // 16: nose cone tip
    pts.push_back(Vec3f(cx, cy, -6.5));

    // 17-20: fin 1 (front, attached at octagon vertex 0)
    float fin_out = 1.8;
    pts.push_back(Vec3f(cx + r,       cy, -0.5));   // 17: base bottom
    pts.push_back(Vec3f(cx + fin_out,  cy, -0.5));   // 18: outer bottom
    pts.push_back(Vec3f(cx + fin_out,  cy, -1.5));   // 19: outer top
    pts.push_back(Vec3f(cx + r,        cy, -2.0));   // 20: base top

    // 21-24: fin 2 (left, attached at octagon vertex 2)
    pts.push_back(Vec3f(cx, cy + r,       -0.5));   // 21
    pts.push_back(Vec3f(cx, cy + fin_out, -0.5));   // 22
    pts.push_back(Vec3f(cx, cy + fin_out, -1.5));   // 23
    pts.push_back(Vec3f(cx, cy + r,       -2.0));   // 24

    // 25-28: fin 3 (back, attached at octagon vertex 4)
    pts.push_back(Vec3f(cx - r,       cy, -0.5));   // 25
    pts.push_back(Vec3f(cx - fin_out, cy, -0.5));   // 26
    pts.push_back(Vec3f(cx - fin_out, cy, -1.5));   // 27
    pts.push_back(Vec3f(cx - r,       cy, -2.0));   // 28

    // 29-32: fin 4 (right, attached at octagon vertex 6)
    pts.push_back(Vec3f(cx, cy - r,       -0.5));   // 29
    pts.push_back(Vec3f(cx, cy - fin_out, -0.5));   // 30
    pts.push_back(Vec3f(cx, cy - fin_out, -1.5));   // 31
    pts.push_back(Vec3f(cx, cy - r,       -2.0));   // 32

    // 33-36: window (diamond shape on the body, front-facing)
    pts.push_back(Vec3f(cx + r + 0.01, cy, -3.5));  // 33: right
    pts.push_back(Vec3f(cx + r + 0.01, cy + 0.4, -3.8)); // 34: top
    pts.push_back(Vec3f(cx + r + 0.01, cy, -4.1));  // 35: left
    pts.push_back(Vec3f(cx + r + 0.01, cy - 0.4, -3.8)); // 36: bottom

    // 37-40: exhaust flame tips
    pts.push_back(Vec3f(cx,       cy,       0.8));   // 37: center flame
    pts.push_back(Vec3f(cx + 0.3, cy + 0.3, 0.5));  // 38: side flame 1
    pts.push_back(Vec3f(cx - 0.3, cy + 0.3, 0.5));  // 39: side flame 2
    pts.push_back(Vec3f(cx,       cy - 0.3, 0.5));   // 40: side flame 3

    vector<Point2f> img_pts;
    projectPoints(pts, rvec, tvec, intrinsics.first, intrinsics.second, img_pts);

    // Body (white) - octagon rings and vertical edges
    Scalar body_color(255, 255, 255);
    for (int i = 0; i < 8; i++) {
        int next = (i + 1) % 8;
        line(src, img_pts[i], img_pts[next], body_color, 2);           // bottom ring
        line(src, img_pts[i + 8], img_pts[next + 8], body_color, 2);   // top ring
        line(src, img_pts[i], img_pts[i + 8], body_color, 2);          // verticals
    }

    // Nose cone (red) - lines from top ring to tip
    Scalar nose_color(0, 0, 255);
    for (int i = 0; i < 8; i++) {
        line(src, img_pts[i + 8], img_pts[16], nose_color, 2);
    }

    // Fins (green)
    Scalar fin_color(0, 200, 0);
    for (int f = 0; f < 4; f++) {
        int base = 17 + f * 4;
        line(src, img_pts[base], img_pts[base + 1], fin_color, 2);
        line(src, img_pts[base + 1], img_pts[base + 2], fin_color, 2);
        line(src, img_pts[base + 2], img_pts[base + 3], fin_color, 2);
        line(src, img_pts[base + 3], img_pts[base], fin_color, 2);
    }

    // Window (cyan)
    Scalar window_color(255, 255, 0);
    line(src, img_pts[33], img_pts[34], window_color, 2);
    line(src, img_pts[34], img_pts[35], window_color, 2);
    line(src, img_pts[35], img_pts[36], window_color, 2);
    line(src, img_pts[36], img_pts[33], window_color, 2);

    // Exhaust flames (orange/yellow)
    Scalar flame_color(0, 140, 255);
    // Flames from bottom ring vertices to flame tips
    line(src, img_pts[0], img_pts[37], flame_color, 2);
    line(src, img_pts[2], img_pts[37], flame_color, 2);
    line(src, img_pts[4], img_pts[37], flame_color, 2);
    line(src, img_pts[6], img_pts[37], flame_color, 2);
    line(src, img_pts[1], img_pts[38], flame_color, 1);
    line(src, img_pts[3], img_pts[39], flame_color, 1);
    line(src, img_pts[5], img_pts[40], flame_color, 1);
}
