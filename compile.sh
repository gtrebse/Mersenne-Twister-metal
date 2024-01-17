#!/bin/bash


# Compile the Metal shader to a .air file
xcrun -sdk macosx metal -c Mersenne_twister.metal -o MyLibrary.air
if [ $? -ne 0 ]; then
    echo "Failed to compile Metal shader."
    exit 1
fi

# Create a Metal library from the .air file
xcrun -sdk macosx metallib MyLibrary.air -o default.metallib
if [ $? -ne 0 ]; then
    echo "Failed to create Metal library."
    exit 1
fi

# Compile the C++ files and link with the Metal framework
clang++ -std=c++17 -I./inc -I./metal-cpp -O2  -framework Metal -framework Foundation -framework MetalKit main.cpp ./src/Mersenne_twister.cpp ./src/Helper_functions.cpp
if [ $? -ne 0 ]; then
    echo "Failed to compile and link C++ files."
    exit 1
fi

echo "Build completed successfully."
