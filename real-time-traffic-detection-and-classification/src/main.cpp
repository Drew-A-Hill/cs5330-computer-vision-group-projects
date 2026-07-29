/*
    Drew Hill & Abhi Banda
    Summer 2026
    CS5330 Final Project

    main.cpp

    This is the main entry point into the program. 
*/

#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

const int PATCH = 20;
const int DEPTH_EVERY = 5;


int main(int argc, char *argv[]) {
    // Ensures correct number of arguments are provided.
    if (argc < 2) {
        cout << "ip address required\n ex: ./main 20.20.20.11" << endl;
        return -1;
    }

    // Builds stream address from ip.
    string ip = argv[1];
    string address = "http://" + ip + ":8080/stream.mjpg";

    // Creates the VideoCapture object from the stream.
    VideoCapture cap;
    cap.open(address);

    // Ensures stream is working
    if (!cap.isOpened()) {

        cout << "******************************" << endl;
        cout << "\nCamera was unable to open\n" << endl;
        cout << "******************************" << endl;
        return -1;
    }


    DA2Network da_net("model_fp16.onnx");

    // Initializes a Mat for the frame.
    Mat frame;
    for (;;) {
        cap >> frame;
        if (frame.empty()) {
            cout<< "Frame is empty" << endl;
            return -1;
        }

        imshow("Traffic Tracker", frame);

        char key = waitKey(1);

        if (key == 'q'){
            return 1;
        }

        if (key == 'r') {
            continue;
            
        } else if (key == 'c') {
            continue;

        }
    }

    return 0;
}