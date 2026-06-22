/*
  Drew Hill & Abhiram Banda
  CS 5330 Pattern Recognition & Computer Vision
  Summer 2026

  features.cpp

  Computes a set of features for a specified region given a region map and a region ID.
  Features computed: centroid, axis of least central moment (orientation),
  oriented bounding box, percent filled, and height/width ratio.
*/

#include "features.h"

/*
  Computes translation, scale, and rotation invariant features for a region.

  const cv::Mat &labels - region map where each pixel's value is its region ID (CV_32S).
  int region_id - the ID of the region to compute features for.
  Returns a RegionFeatures struct containing all computed features.
*/
RegionFeatures computeFeatures(const cv::Mat &labels, int region_id){

    // Compute the centroid (center of mass)
    // Sum all x and y coordinates of pixels in the region, then divide by count.
    // The centroid is the average position of all pixels in the region.
    int sum_x = 0;
    int sum_y = 0;
    int count = 0;
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (labels.at<int>(y, x) == region_id){
                sum_x += x;
                sum_y += y;
                count++;
            }
        }
    }
    float centroid_x = (float)sum_x / count;
    float centroid_y = (float)sum_y / count;

    // Compute central moments and collect region points
    // Central moments describe how the region's mass is distributed relative
    // to its centroid. They are used to find the orientation of the region.
    //   mu20: variance in x — how spread out the region is horizontally
    //   mu02: variance in y — how spread out the region is vertically
    //   mu11: covariance of x and y — the correlation between horizontal
    //   and vertical spread (indicates diagonal elongation)
    float mu20 = 0;
    float mu02 = 0;
    float mu11 = 0;
    std::vector<cv::Point> points;
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (labels.at<int>(y, x) == region_id){
                mu20 += (x - centroid_x) * (x - centroid_x);
                mu02 += (y - centroid_y) * (y - centroid_y);
                mu11 += (x - centroid_x) * (y - centroid_y);
                points.push_back(cv::Point(x, y));
            }
        }
    }

    // Compute the axis of least central moment (orientation)
    // This is the angle of the axis through the centroid around which the
    // region has the least rotational inertia — i.e., the axis aligned with
    // the "longest" direction of the shape. Derived from the eigenvectors
    // of the covariance matrix [mu20, mu11; mu11, mu02].
    float angle = 0.5f * atan2(2.0f * mu11, mu20 - mu02);

    // Compute the oriented bounding box
    // minAreaRect finds the smallest rotated rectangle enclosing all region pixels.
    cv::RotatedRect bbox = cv::minAreaRect(points);

    // Compute scale and rotation invariant features
    // Ensure width is the longer side for consistent hw_ratio
    float width = bbox.size.width;
    float height = bbox.size.height;
    if (width < height) std::swap(width, height);

    // Percent filled: ratio of actual region pixels to bounding box area.
    // A circle would be ~0.78, a rectangle ~1.0, a thin cross much less.
    float percent_filled = (float)count / (bbox.size.width * bbox.size.height);

    // Height/width ratio: always <= 1 since height is the shorter side.
    // A square or circle gives ~1.0, an elongated shape gives a small value.
    float hw_ratio = height / width;

    // Pack results into struct
    RegionFeatures features;
    features.centroid_x = centroid_x;
    features.centroid_y = centroid_y;
    features.angle = angle;
    features.oriented_bbox = bbox;
    features.percent_filled = percent_filled;
    features.hw_ratio = hw_ratio;

    return features;
}
