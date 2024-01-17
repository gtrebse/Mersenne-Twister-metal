/*
A class to abstract the Mersenne Twister random number generator
*/

#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

//TODO: Make rngPerThread a parameter to optimize GPU use
//TODO: Make arrayLength a  parameter
//#define gridSize 1024
//#define arrayLength 1024*1e5
//#define rngPerThread arrayLength/gridSize
//#define bufferSize arrayLength * sizeof(unsigned char)
#define MT_NN 19
#define MT_WMASK 0xFFFFFFFFU

// Struct to hold Mersenne Twister parameters
typedef struct{
  unsigned int matrix_a;
  unsigned int mask_b;
  unsigned int mask_c;
  unsigned int seed;
} mt_struct_stripped;

class MetalMT
{
public:
    MTL::Device *_mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    MTL::ComputePipelineState *_mAddFunctionPSO;

    // The command queue used to pass commands to the device.
    MTL::CommandQueue *_mCommandQueue;

    MTL::CommandBuffer *_mCommandBuffer;

    MTL::ComputeCommandEncoder *_mComputeCommandEncoder;

    // Buffers to hold data.
    MTL::Buffer *_d_MT;
    MTL::Buffer *_MT_state;
    MTL::Buffer *_n_rng_per_thread;
    MTL::Buffer *_n_threads;
    MTL::Buffer *_mResultB;

    MetalMT(MTL::Device *device, int seed = 1);
    ~MetalMT();

    void prepareData();
    void sendComputeCommand(int arrayLenght);
    void setSeed(int seed);
    
    //Function to return result in a 2d vector
    std::vector<unsigned char> getResult();
    void printResult();

private:
    int _seed;
    int _rngPerThread;
    int _max_available_threads;
    int _arrayLength;
    unsigned int *_MT_state_buffer;

    void encodeBernoulliCommand(MTL::ComputeCommandEncoder *commandEncoder);
};