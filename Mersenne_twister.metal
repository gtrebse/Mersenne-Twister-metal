/*

Abstract:
A shader that creates Bernoulli random variables
*/

#include <metal_stdlib>
using namespace metal;
/// This is a Metal Shading Language (MSL) function equivalent to the create Bernoulli random variables

//Struct to hold MT parameters
typedef struct{
  unsigned int matrix_a;
  unsigned int mask_b;
  unsigned int mask_c;
  unsigned int seed;
} mt_struct_stripped;

#define     MT_MM 9
#define     MT_NN 19
#define     MT_WMASK 0xFFFFFFFFU
#define     MT_UMASK 0xFFFFFFFEU
#define     MT_LMASK 0x1U
#define     MT_SHIFT0 12
#define     MT_SHIFTB 7
#define     MT_SHIFTC 15
#define     MT_SHIFT1 18
#define     PI 3.14159265358979f                            

kernel void bernoulli_kernel(device const mt_struct_stripped* d_MT,
                             device const int* n_rng_per_thread,
                             device const int* n_threads,
                             device unsigned char* result,
                             device unsigned int* MT_state,
                             uint index [[thread_position_in_grid]],
                             uint index_in_threadgroup [[thread_position_in_threadgroup]])
{
    //Create local states
    unsigned int threshold = (unsigned int)(0.5 * 4294967295);
    int iState, iState1, iStateM, iOut, _nThreads, _nRngPerThread;

    //unsigned char buffer;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN], _matrix_a, _mask_b, _mask_c;

    //Load bit-vector MT parameters
    _matrix_a       = d_MT[index].matrix_a;
    _mask_b         = d_MT[index].mask_b;
    _mask_c         = d_MT[index].mask_c;

    //Load GPU states
    _nThreads       = *n_threads;
    _nRngPerThread  = *n_rng_per_thread;

    for(iState = 0; iState < MT_NN; iState++){
        mt[iState] = MT_state[index * (MT_NN + 1) + iState];
    }

    //Initialize states
    iState = MT_state[index * (MT_NN + 1) + MT_NN];
    mti1 = mt[0];

    //Generate _nRngPerThread random numbers
    for(iOut = 0; iOut < _nRngPerThread; iOut++){
        iState1 = iState + 1;
        iStateM = iState + MT_MM;
        if(iState1 >= MT_NN) iState1 -= MT_NN;
        if(iStateM >= MT_NN) iStateM -= MT_NN;
        mti  = mti1;
        mti1 = mt[iState1];
        mtiM = mt[iStateM];

        //  MT recurrence
        x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
        x =  mtiM ^ (x >> 1) ^ ((x & 1) ? _matrix_a : 0);

        mt[iState] = x;
        iState = iState1;

        //Tempering transformation
        x ^= (x >> MT_SHIFT0);
        x ^= (x << MT_SHIFTB) & _mask_b;
        x ^= (x << MT_SHIFTC) & _mask_c;
        x ^= (x >> MT_SHIFT1);

        result[index + _nThreads * iOut] = (unsigned char) (x < threshold) ? 1 : 0;
    }

    //Save current mt state
    //Unrolled loop
    MT_state[index * 20 + 0] = mt[0];
    MT_state[index * 20 + 1] = mt[1];
    MT_state[index * 20 + 2] = mt[2];
    MT_state[index * 20 + 3] = mt[3];
    MT_state[index * 20 + 4] = mt[4];
    MT_state[index * 20 + 5] = mt[5];
    MT_state[index * 20 + 6] = mt[6];
    MT_state[index * 20 + 7] = mt[7];
    MT_state[index * 20 + 8] = mt[8];
    MT_state[index * 20 + 9] = mt[9];
    MT_state[index * 20 + 10] = mt[10];
    MT_state[index * 20 + 11] = mt[11];
    MT_state[index * 20 + 12] = mt[12];
    MT_state[index * 20 + 13] = mt[13];
    MT_state[index * 20 + 14] = mt[14];
    MT_state[index * 20 + 15] = mt[15];
    MT_state[index * 20 + 16] = mt[16];
    MT_state[index * 20 + 17] = mt[17];
    MT_state[index * 20 + 18] = mt[18];

    // The last part of the original loop, setting the iState
    MT_state[index * 20 + 19] = iState;
}