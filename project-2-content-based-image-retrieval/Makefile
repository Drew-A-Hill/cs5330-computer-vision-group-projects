CXX = g++
CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)

csv_helper: csv_helper.cpp csv_util.cpp features.cpp features_helper.cpp
	$(CXX) -o csv_helper csv_helper.cpp csv_util.cpp features.cpp features_helper.cpp $(CXXFLAGS) $(LDFLAGS)

matcher: matcher.cpp csv_util.cpp features.cpp features_helper.cpp
	$(CXX) -o matcher matcher.cpp csv_util.cpp features.cpp features_helper.cpp $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f csv_helper matcher