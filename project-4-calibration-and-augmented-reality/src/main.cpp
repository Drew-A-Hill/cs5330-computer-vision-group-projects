/*
  Drew Hill & Abhiram Banda,
  Summer 2026
  CS 5330 Computer Vision

  src/main.cpp

  Entry into the program.
*/

#include <opencv2/opencv.hpp>
#include <string>
#include "action_runner.h"

using namespace cv;
using namespace std;

/*
  Handles the input arguments from the command line and launches the desired action. 
  Use ./main action input_type {IP, file name or none}
  Actions: 
  "c" runs calibrate, then immediately run AR with the new intrinsics.
  "ar" run AR using previously saved intrinsics.

  Input type:
  User can use mobile camera using an IP stream app, webcam, prerecorded video, or image.
  "mobile" uses a mobile ip streaming app as the video source. Requires IP to run ex. 1.22.13.1
  "web" uses the users web cam and does not require any further arguments.
  "video" loads a video and to run the program. Requires the file path to the video.
  "img loads a img and to run the program. Requires the file path to the image.
*/
int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "More arguments required: ./main {c|ar} {input args}" << endl;
        return 1;
    }

    string action = argv[1];
    string intrinsics_path = "./data/intrinsic_parameters.yml";
    vector<string> args;

    // Collect everything after the program name and mode into the args list.
    for (int i = 0; i < argc; i++) {
        if (i < 2) {
            continue;
        }
        
        args.push_back(argv[i]);
    }

    // Holds orientation of the camera.
    char orientation = '0';
    if (action == "c") {
        // Calibrate first, then run AR reusing the orientation that was set.
        calibration_runner(args, orientation);
        ar_runner(intrinsics_path, args, orientation);
        cout << "\n\nORIENTATION: " << orientation << endl;

    } else if  (action == "ar") {
        // Skip calibration and run AR straight from the saved intrinsics file.
        ar_runner(intrinsics_path, args, orientation);

    }

    return 0;
}
    