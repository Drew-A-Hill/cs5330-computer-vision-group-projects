CXX = g++
CXXFLAGS = -std=c++17 -O2 $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)

SRC = src/main.cpp src/calibration.cpp src/action_runner.cpp src/ar.cpp

main: $(SRC)
	$(CXX) -o main $(SRC) $(CXXFLAGS) $(LDFLAGS)

features: src/features.cpp
	$(CXX) -o features src/features.cpp $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f main features