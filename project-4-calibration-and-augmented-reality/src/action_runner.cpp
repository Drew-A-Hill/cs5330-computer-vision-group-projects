/*
  Drew Hill & Abhiram Banda,
  Summer 2026
  CS 5330 Computer Vision

  src/action_runner.cpp

  Program which holds the two run modes: calibration_runner, amd ar_runner.
  calibration_runner collects chessboard views, computes, and saves the camera 
  intrinsics. ar_runner loads those intrinsics to overlay object and 3D axes on 
  the target.
*/

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "calibration.h"
#include "ar.h"

using namespace cv;
using namespace std;

/*
    Runs the calibration. Opens the selected camera source, then loops detecting the 9x6 
    chessboard each frame. 
    Keys 0-3: set the view rotation. 
    's': saves the current detection. 
    'c': calibrates once 5+ views are saved.
    'w': writes intrinsics and orientation to file. 
    'q': quits the program.
    
    std::vector<string> args - input type followed by any source argument.
    char &orientation - view rotation used.
    returns 1 on normal exit, -1 on camera or frame failure.
*/
int calibration_runner(vector<string> args, char &orientation) {
    VideoCapture cap;
    Mat img;

    string input_type = args[0];

    // Selects the frame source of live webcam, mobile, video, or image.
    if (input_type == "web") {
        cap.open(0);

        if (!cap.isOpened()) {
            printf("Web cam unable to open");
            return -1;
        }

    } else if (input_type == "mobile") {
        // Buids the stream URL from mobile devices IP and opens VideoCapture.
        string ip = args[1];
        string url = "http://" + ip + ":8080/stream.mjpg";
            
        cap.open(url);

        if (!cap.isOpened()) {
            printf("Mobile camera unable to open");
            return -1;
        }

    } else {
        // Creates file path from arguments
        string dir = "../";
        string file_name = args[1];
        string file_path = dir + file_name;

        if (input_type == "vid") {
            img = imread(file_path);

        } else if (input_type == "img") {
            cap.open(file_path);

        } else {
            // Prints error message if wrong arguments are being pased.
            cout << "\nError:\n" 
                 << input_type
                 << " is not a valid action selection. " 
                 << "Requires use of one of the following action: web, mobile, vid, img. " 
                 << "\nExample: ./main mobile 0.11.23.1\n"
                 << endl;
        }
    }
    
    Mat frame;
    Mat dst;
    
    // 3D points per saved view.
    vector<vector<Vec3f>> saved_points_list;

    // Corners per saved view.
    vector<vector<Point2f>> saved_corners_list;

    int wait_val;

    // Current rotation state.
    char rotations = '0';

    // Holds camera_matrix and distortion.
    pair<Mat, vector<double>> intrinsics;
    for (;;) {
        
        // Gets the frame if not an image
        if (input_type != "img") {
            cap >> frame;
            wait_val = 1;

        } else {
            frame = img.clone();
            wait_val = 1;

        }

        // Checks if frame is empty.
        if (frame.empty()) {
            cout << "Frame is empty\n" << endl;
            return -1;
        }

        dst = frame;

        // Applies rotation if selected by user.
        if (rotations == '1') {
            rotate(dst, dst, ROTATE_90_CLOCKWISE);

        } else if (rotations == '2') {
            rotate(dst, dst, ROTATE_180);

        } else if (rotations == '3') {
            rotate(dst, dst, ROTATE_90_COUNTERCLOCKWISE);

        }

        // Sets size of corners.
        Size size(9, 6);
        vector<Point2f> corners;

        // Detects if chessboard detected in chessboard of set size. If detected returns true esle returns false.
        bool detected = detect_corners(dst, size, corners);
        drawChessboardCorners(dst, size, corners, detected);

        // Builds the matching 3D points for the board.
        vector<Vec3f> saved_point_set = point_set(size);

        // Holds the corner count and first corner for the summary printout after quit.
        Point2f first_corner(0, 0);
        int corner_count = 0;
        if (!corners.empty()) {
            first_corner = corners[0];
            corner_count = corners.size();
        }

        // Displays window
        imshow("Detect", dst);

        char key = waitKey(wait_val);
        

        if (key == 'q') {
            if (corners.empty() && corner_count == 0) {
                cout << "No Corneres Detected" << endl;
            }
            destroyWindow("Detect");

            cout << "\nFirst Corner: " << first_corner 
            << "\nNumber of Corners " << corner_count << endl;

            return 1;
        } 

        // Handles rotation selection.
        if (key == '1') {
            rotations = '1';

        } else if (key == '2') {
            rotations = '2';

        } else if (key == '3') {
            rotations = '3';

        } else if (key == '0') {
            rotations = '0';

        } else if(key == 's') {
            // Save the latest detection's corners and matching 3D points as one calibration view.
            saved_points_list = point_list(saved_points_list, saved_point_set, detected);
            saved_corners_list = corner_list(saved_corners_list, corners, detected);

            cout << "Saved:" << saved_corners_list.size() << endl;
        } else if(key == 'c') {
            // Calibrate once at least 5 views have been collected.
            if (saved_corners_list.size() > 4) {
                intrinsics = calibrate_camera(dst, saved_corners_list, saved_points_list);

            } else {
                cout << "Minimum 5 corner data points required. You currently have: " << saved_corners_list.size() << endl;
            }

        } else if(key == 'w') {
            // Saves intrinsics and orientation so ar runner can reuse them without recalibrating.
            if (!intrinsics.first.empty() && !intrinsics.second.empty()) {
                string intrinsics_filename = "./data/intrinsic_parameters.yml";
                FileStorage write(intrinsics_filename, FileStorage::WRITE);
                if (write.isOpened()) {
                    cout << "Writting write intrinsics" << endl;
                    write << "camera_matrix" << intrinsics.first;
                    write << "distortion" << intrinsics.second;
                    cout << "Intrinsics written to " << intrinsics_filename << endl;
                    write << "orientation" << string(1, rotations); 
                    orientation = rotations;

                } else {
                    cout << "Failed to write intrinsics" << endl;

                }
            } else {
                cout << "Intrinsics not found" << endl;
            }
        }
    }
    return 1;
}

/*
    Runs the ar. Loads the saved intrinsics and orientation, then loops detecting the target and computing its pose.
    When a valid pose is found, projects and draws the 3D coordinate axes onto the board. Applies the same rotation 
    used during calibration.

    std::string intrinsics_path - path to the saved intrinsics file.
    std::vector<string> args - input type followed by any source argument.
    char orientation - view rotation to match the calibration.
    returns 1 on normal exit, -1 on camera or frame failure.
*/
int ar_runner(string intrinsics_path, vector<string> args, char orientation) {
    VideoCapture cap;
    Mat img;

    string input_type = args[0];

    // Selects the frame source of live webcam, mobile, video, or image.
    if (input_type == "web") {
        cap.open(0);

        if (!cap.isOpened()) {
            printf("Web cam unable to open");
            return -1;
        }

    } else if (input_type == "mobile") {
        string ip = args[1];
        string url = "http://" + ip + ":8080/stream.mjpg";
            
        cap.open(url);

        if (!cap.isOpened()) {
            printf("Mobile camera unable to open");
            return -1;
        }

    } else {
        // Creates file path from arguments
        string dir = "../";
        string file_name = args[1];
        string file_path = dir + file_name;

        if (input_type == "vid") {
            cap.open(file_path);

        } else if (input_type == "img") {
            img = imread(file_path);

        } else {
            // Prints error message if wrong arguments are being pased.
            cout << "\nError:\n" 
                 << input_type
                 << " is not a valid action selection. " 
                 << "Requires use of one of the following action: web, mobile, vid, img. " 
                 << "\nExample: ./main mobile 0.11.23.1\n"
                 << endl;
        }
    }

    Mat frame;
    Mat dst;

    int wait_val;

    // Current rotation state.
    char rotations = '0';

    // Path to intrinsics.
    string intrinsics_filename = "./data/intrinsic_parameters.yml";

    // Reads the previously computed camera_matrix and distortion from file.
    pair<Mat, vector<double>> intrinsics = read_intrinsics(intrinsics_filename, orientation);
    
    if (intrinsics.first.empty() || intrinsics.second.empty()) {
        cout << "No intrinsics found. You must calibrate before runing ar" << endl;

    }

    // Sets size of corners.
    Size size(9, 6);

    // Builds the matching 3D points for the board.
    vector<Vec3f> saved_point_set = point_set(size);

    for (;;) {
        // Gets the frame if not an image
        if (input_type != "img") {
            cap >> frame;
            wait_val = 1;

        } else {
            frame = img.clone();
            wait_val = 1;

        }

        if (frame.empty()) {
            cout << "Frame is empty\n" << endl;
            return -1;
        }

        dst = frame;

        // Applies rotation if selected by user during calibration.
        if (orientation == '1') {
            rotate(dst, dst, ROTATE_90_CLOCKWISE);

        } else if (orientation == '2') {
            rotate(dst, dst, ROTATE_180);

        } else if (orientation == '3') {
            rotate(dst, dst, ROTATE_90_COUNTERCLOCKWISE);

        }

        vector<Point2f> corners;
        
        Mat rvec;
        Mat tvec;

        // Computes the chessbpoards pose for the frame.
        bool computed = compute_pose(dst, size, saved_point_set, corners, intrinsics, rvec, tvec);

        // Draws axis if computed
        if (computed) {
            cout << "rvec: " << rvec << "\ntvec: " << tvec << endl;
            draw_axis(dst, rvec, tvec, intrinsics);
        }
        
        // Displays window.
        imshow("AR", dst);
        char key = waitKey(wait_val);

        if (key == 'q') {
            return 1;
        }
    }

    return 1;
}