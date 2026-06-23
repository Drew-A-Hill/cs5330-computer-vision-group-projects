CXX = g++
# CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4)
CXXFLAGS = -std=c++17 -O2 `pkg-config --cflags opencv4`
LDFLAGS = $(shell pkg-config --libs opencv4)

SRC = src/main.cpp src/threshold.cpp src/morpholocial-filter.cpp src/segmenting.cpp src/features.cpp src/training.cpp src/csv_util.cpp src/classifier.cpp

main: $(SRC) src/threshold.h
	$(CXX) -o main $(SRC) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f main
