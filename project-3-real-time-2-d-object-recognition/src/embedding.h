/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  embedding.h

  Header file for the one-shot embedding-based classification system.
  Computes a ResNet18 embedding for a region and classifies it against a
  database of known embeddings using cosine distance.
*/

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "features.h"

using namespace cv;
using namespace std;

// Holds one labeled embedding example in the database.
struct EmbeddingEntry {
    string label;
    Mat embedding;
};
void compute_axis_extents(const Mat &labels, int region_id, const RegionFeatures &features, float &min_e1, float &max_e1, float &min_e2, float &max_e2);
Mat compute_region_embedding(Mat &frame, const Mat &labels, int region_id, const RegionFeatures &features, dnn::Net &net);
string classify_embedding(Mat &unknown, vector<EmbeddingEntry> &db, float threshold);
int save_embedding(const string &label, Mat &embedding, bool append);
vector<EmbeddingEntry> load_embeddings(const string &filename);

#endif