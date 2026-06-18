CXX = g++
CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)

SRC = src/main.cpp src/threshold.cpp

main: $(SRC) src/threshold.h
	$(CXX) -o main $(SRC) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f main
