/*
  Drew Hill & Abhiram Banda,
  Summer 2026
  CS 5330 Computer Vision

  src/action_runner.h

  Header file for the action_runner.cpp.
*/

#ifndef ACTION_RUNNER_H
#define ACTION_RUNNER_H

#include <vector>
using namespace std;

int calibration_runner(vector<string> args, char &orientation);
int ar_runner(string intrinsics_path, vector<string> args, char orientation);

#endif