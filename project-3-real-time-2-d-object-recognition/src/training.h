#ifndef TRAINING_H
#define TRAINING_H

#include <string>
#include "features.h"

int saveTrainingData(const std::string &label, const RegionFeatures &features, bool append);

#endif
