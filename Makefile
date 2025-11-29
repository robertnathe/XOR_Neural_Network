CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra
INCLUDES = -I/usr/include/eigen3 -I/usr/include
LIBS = -lboost_system -lboost_filesystem -lboost_math_c99 -lcurl
TARGET = xor_nn_fixed_size_optimized
SOURCE = xor_nn_fixed_size_optimized.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

.PHONY: clean

clean:
	rm -f $(TARGET) *.o
