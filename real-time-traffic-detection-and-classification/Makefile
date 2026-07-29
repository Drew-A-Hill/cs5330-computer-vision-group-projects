CXX = g++
ONNX = $(shell brew --prefix onnxruntime)
CXXFLAGS = -std=c++17 -O2 $(shell pkg-config --cflags opencv4) -I$(ONNX)/include/onnxruntime
LDFLAGS = $(shell pkg-config --libs opencv4) -L$(ONNX)/lib -lonnxruntime

SRC = src/main.cpp
main: $(SRC)
	$(CXX) -o main $(SRC) $(CXXFLAGS) $(LDFLAGS)