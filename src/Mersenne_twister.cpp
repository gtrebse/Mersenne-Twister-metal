/*
A class to abstract the Mersenne Twister random number generator
*/

#include "Mersenne_twister.hpp"
#include <iostream>


MetalMT::MetalMT(MTL::Device *device, int seed): _seed(seed){

    _mDevice = device;
    //_seed = seed;

    NS::Error *error = nullptr;

    // Load the shader files with .metal extension
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();

    if(defaultLibrary == nullptr){
        std::cout << "Failed to load default library." << std::endl;
        return;
    }

    auto bernoulli_kernel_str = NS::String::string("bernoulli_kernel", NS::ASCIIStringEncoding);
    MTL::Function *bernoulliFunction = defaultLibrary->newFunction(bernoulli_kernel_str);

    if(bernoulliFunction == nullptr){
        std::cout << "Failed to load bernoulli function." << std::endl;
        return;
    }

    // Create a compute pipeline state object
    _mAddFunctionPSO = _mDevice->newComputePipelineState(bernoulliFunction, &error);
    bernoulliFunction->release();

    if(_mAddFunctionPSO == nullptr){
        std::cout << "Failed to create compute pipeline state, error" << error << std::endl;
        return;
    }

    // Get max number of threads per threadgroup
    _max_available_threads = _mAddFunctionPSO->maxTotalThreadsPerThreadgroup();

    // Allocate the _MT_state_buffer based on available threads
    _MT_state_buffer = new unsigned int[_max_available_threads * (MT_NN + 1) * 100];

    // Create a command queue
    _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr){
        std::cout << "Failed to create command queue." << std::endl;
        return;
    }

    //Allocate buffers to hold initial data and the results
    _d_MT =             _mDevice->newBuffer(sizeof(mt_struct_stripped)*_max_available_threads, MTL::ResourceStorageModeShared);
    _MT_state =         _mDevice->newBuffer(sizeof(unsigned int) * _max_available_threads * (MT_NN + 1) * 100, MTL::ResourceStorageModeShared);
    _n_rng_per_thread = _mDevice->newBuffer(sizeof(unsigned int), MTL::ResourceStorageModeShared);
    _n_threads =        _mDevice->newBuffer(sizeof(unsigned int), MTL::ResourceStorageModeShared);

    std::cout << "Allocation done, preparing data." << std::endl;
    prepareData();
}

void MetalMT::prepareData(){

    // The pointer needs to be explicitly cast in C++
    mt_struct_stripped *mt_ptr = static_cast<mt_struct_stripped *>(_d_MT->contents());
    mt_struct_stripped *mt = new mt_struct_stripped[_max_available_threads];

    FILE *fp;
    fp = fopen("./data/MersenneTwister.dat", "rb");

    if(fp == nullptr){
        std::cout << "Failed to open file." << std::endl;
        return;
    }

    // Read data from source file
    size_t readCount;
    
    for(int i = 0; i < _max_available_threads; ++i){
        readCount = fread(&mt[i], sizeof(mt_struct_stripped), 1, fp);
        if(readCount != 1){
            if(feof(fp)){ // If end of file is reached
                fseek(fp, 0, SEEK_SET); // Seek back to the start of the file

                clearerr(fp);
                // Try reading again
                if(fread(&mt[i], sizeof(mt_struct_stripped), 1, fp) != 1){
                    std::cout << "Failed to read data after seeking back to start for index " << i << std::endl;
                    delete[] mt;
                    fclose(fp);
                    return;
                }
            } else {
                // If fread failed for reasons other than EOF, print error and return
                std::cout << "Failed to read data for index " << i << std::endl;
                delete[] mt;
                fclose(fp);
                return;
            }
        }
    }

    //Close the file
    fclose(fp);

    //Copy the data into the buffer variables
    memcpy(mt_ptr, mt, sizeof(mt_struct_stripped)*_max_available_threads);

    setSeed(_seed);

    //Free the memory
    delete[] mt;
}


void MetalMT::setSeed(int seed){
    // The pointer needs to be explicitly cast in C++
    _seed = seed;

    int j;
    for(int i = 0; i < _max_available_threads*100; ++i){
        j = 0;
        _MT_state_buffer[i * (MT_NN + 1) + j] = i + _seed*_max_available_threads;
        for(j = 1; j < MT_NN; j++){
           _MT_state_buffer[(i%_max_available_threads) * (MT_NN + 1) + j] = (1812433253U * (_MT_state_buffer[i*(MT_NN + 1) + j - 1] ^ (_MT_state_buffer[i*(MT_NN + 1) + j - 1] >> 30)) + j) & MT_WMASK;
        }
        _MT_state_buffer[(i%_max_available_threads) * (MT_NN + 1) + MT_NN] = 0;
   }
   //Save new initial state
   memcpy(static_cast<unsigned int *>(_MT_state->contents()), _MT_state_buffer, sizeof(unsigned int) * _max_available_threads * (MT_NN + 1)*100);
}


void MetalMT::sendComputeCommand(int arrayLength){
    //Set the array length
    _arrayLength = arrayLength;

    //Create command buffer to hold commands
    _mCommandBuffer = _mCommandQueue->commandBuffer();
    assert(_mCommandBuffer != nullptr);

    //Start compute pass
    _mComputeCommandEncoder = _mCommandBuffer->computeCommandEncoder();
    assert(_mComputeCommandEncoder != nullptr);

    encodeBernoulliCommand(_mComputeCommandEncoder);

    //End compute pass
    _mComputeCommandEncoder->endEncoding();

    //Execute the command
    _mCommandBuffer->commit();
 
    //Wait for the command to finish executing, not best practice, as it blocks app while GPU is running
    _mCommandBuffer->waitUntilCompleted();
    
    //Release the command buffer
    _mCommandBuffer->release();

    //Release the command encoder
    _mComputeCommandEncoder->release();

}

void MetalMT::encodeBernoulliCommand(MTL::ComputeCommandEncoder *computeEncoder){
    //Calculate threadgroup size and number of vars per thread based on array length
    int threadgroupSize = _max_available_threads*100;
    int numVarsPerThread = std::ceil(static_cast<double>(_arrayLength) / threadgroupSize);

    //Set the compute pipeline state object for the encoder
    computeEncoder->setComputePipelineState(_mAddFunctionPSO);

    //Set the threadgroup size
    MTL::Size GPUGridSize = MTL::Size::Make(threadgroupSize, 1, 1);

    //Assign buffer to hold the number of vars per thread
    memcpy(_n_rng_per_thread->contents(), &numVarsPerThread, sizeof(numVarsPerThread));
    memcpy(_n_threads->contents(), &threadgroupSize, sizeof(_max_available_threads));
    
    //Set the threadgroup size
    MTL::Size threadGroupSize_inp = MTL::Size::Make(128, 1, 1);

    //Allocate buffer to hold the results
    _mResultB = _mDevice->newBuffer(sizeof(unsigned char)*numVarsPerThread*threadgroupSize, MTL::ResourceStorageModeShared);

    //Set the buffers
    computeEncoder->setBuffer(_d_MT, 0, 0);
    computeEncoder->setBuffer(_n_rng_per_thread, 0, 1);
    computeEncoder->setBuffer(_n_threads, 0, 2);
    computeEncoder->setBuffer(_mResultB, 0, 3);
    computeEncoder->setBuffer(_MT_state, 0, 4);

    //Encode the compute command
    computeEncoder->dispatchThreads(GPUGridSize, threadGroupSize_inp);
}

void MetalMT::printResult(){

    long long cumulative_sum = 0;

    unsigned char *result_ptr = (unsigned char *)_mResultB->contents();
    std::cout << "Compute done, analysing results." << std::endl;
    for (unsigned long long i = 0; i < _arrayLength; ++i){
        cumulative_sum += result_ptr[i];
    }
    std::cout << "Cumulative sum: " << cumulative_sum << std::endl;
    std::cout << "Number of vars: " << _arrayLength << std::endl;
    std::cout << "Cumulative average: " << static_cast<double>(cumulative_sum) / (_arrayLength) << std::endl;
}

std::vector<unsigned char> MetalMT::getResult(){
    std::vector<unsigned char> result;
    unsigned char *result_ptr = (unsigned char *)_mResultB->contents();
    for (unsigned long long i = 0; i < _arrayLength; ++i){
        result.push_back(result_ptr[i]);
        //std::cout << result_ptr[i] << std::endl;
    }

    //std::vector<unsigned char> result;
    //result.reserve(_arrayLength); // Preallocate memory for the number of elements
    //unsigned char *result_ptr = (unsigned char *)_mResultB->contents();

    //// Copy the bytes using std::copy
    //std::copy(result_ptr, result_ptr + _arrayLength, std::back_inserter(result));
    
    //unsigned int *state_ptr = (unsigned int *)_MT_state->contents();
    // Print first 19 elements in vecotr *_MT_STATE_BUFFER
    //for (int i = 0; i < 20; ++i){
    //    std::cout << state_ptr[i] << " ";
    //}

    return result;
}


MetalMT::~MetalMT(){
    _mAddFunctionPSO->release();
    _mCommandQueue->release();

    _d_MT->release();
    _MT_state->release();
    _n_rng_per_thread->release();
    _n_threads->release();
    _mResultB->release();
    _mDevice->release();

    delete[] _MT_state_buffer;
}