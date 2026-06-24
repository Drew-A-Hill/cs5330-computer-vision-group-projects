/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  embedding.cpp

  Embedding based classifier. Crops a region to an upright orientation, runs it throug
  ResNet18, and matches it against saved embeddings using cosine distance. 
*/

#include "embedding.h"
#include "vision.h"
#include <fstream>
#include <sstream>
#include <limits>

using namespace cv;
using namespace std;

/*
  Finds how far the region stretches in each direction. Looks at every pixel in the region 
  and measures its distance along two directions. The smallest and largest distances become 
  the edges of the crop box. 

  const Mat &labels - Region map where each pixel holds its region ID.
  int region_id - Which region to measure.
  const RegionFeatures &features - The centroid and angle.
  float &min_e1, &max_e1 - How far it reaches along the long axix.
  float &min_e2, &max_e2 - How far it reaches along the short axis.
*/
void compute_axis_extents(const Mat &labels, int region_id, const RegionFeatures &features, float &min_e1, float &max_e1, float &min_e2, float &max_e2) {
    min_e1 = numeric_limits<float>::max();
    max_e1 = -numeric_limits<float>::max();
    min_e2 = numeric_limits<float>::max();
    max_e2 = -numeric_limits<float>::max();

    float ct = cos(features.angle);
    float st = sin(features.angle);

    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (labels.at<int>(y, x) != region_id) continue;

            float dx = x - features.centroid_x;
            float dy = y - features.centroid_y;

            // project the pixel onto each axis
            float proj1 = dx * ct + dy * st;
            float proj2 = -dx * st + dy * ct;

            if (proj1 < min_e1) min_e1 = proj1;
            if (proj1 > max_e1) max_e1 = proj1;
            if (proj2 < min_e2) min_e2 = proj2;
            if (proj2 > max_e2) max_e2 = proj2;
        }
    }
}


/*

*/
Mat compute_region_embedding(Mat &frame, const Mat &labels, int region_id, const RegionFeatures &features, dnn::Net &net) {
    float min_e1, max_e1, min_en2, max_e2;
    compute_axis_extents(labels, region_id, features, min_e1, max_e1, min_en2, max_e2);

    // straighten and crop the region
    Mat emb_image;
    prepEmbeddingImage(frame, emb_image, (int)features.centroid_x, (int)features.centroid_y, features.angle, min_e1, max_e1, min_en2, max_e2, 0);

    // run it through the network
    Mat embedding;
    getEmbedding(emb_image, embedding, net, 0);

    return embedding.clone();
}

// Cosine distance between two vectors (1 - cosine similarity). 0 means equal.
float cosine_distance(Mat &a, Mat &b) {
    double dot = a.dot(b);
    double norm_a = norm(a);
    double norm_b = norm(b);

    if (norm_a == 0 || norm_b == 0) return 1.0f;

    return (float)(1.0 - (dot / (norm_a * norm_b)));
}

string classify_embedding(Mat &unknown, vector<EmbeddingEntry> &db, float threshold) {
    float min_dist = numeric_limits<float>::max();
    string selected_label = "Unknown";

    for (int i = 0; i < db.size(); i++) {
        float dist = cosine_distance(unknown, db[i].embedding);
        if (dist < min_dist) {
            min_dist = dist;
            selected_label = db[i].label;
        }
    }

    if (min_dist > threshold) {
        return "Unknown";
    }

    return selected_label;
}

/*
  
*/
int save_embedding(const string &label, Mat &embedding, bool append) {
    ofstream outfile;
    if (append)
        outfile.open("embedding_data.csv", ios_base::app);
    else
        outfile.open("embedding_data.csv");

    if (!outfile.is_open()) {
        cerr << "Error: could not open embedding_data.csv" << endl;
        return -1;
    }

    outfile << label;

    // dump each value after the label
    Mat flat = embedding.reshape(1, 1);
    for (int i = 0; i < flat.cols; i++) {
        outfile << "," << flat.ptr<float>(0)[i];
    }
    outfile << "\n";

    outfile.close();
    return 0;
}

/*

*/
vector<EmbeddingEntry> load_embeddings(const string &filename) {
    vector<EmbeddingEntry> db;
    ifstream infile(filename);

    if (!infile.is_open()) {
        cerr << "Error: could not open " << filename << endl;
        return db;
    }

    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        string token;
        EmbeddingEntry entry;

        // label comes first
        getline(ss, token, ',');
        entry.label = token;

        // everything after it is the vector
        vector<float> values;
        while (getline(ss, token, ',')) {
            values.push_back(stof(token));
        }

        // keep it as a 1-row float Mat so cosine_distance can take it
        entry.embedding = Mat(1, values.size(), CV_32F);
        for (int i = 0; i < values.size(); i++) {
            entry.embedding.at<float>(0, i) = values[i];
        }

        db.push_back(entry);
    }

    infile.close();
    return db;
}