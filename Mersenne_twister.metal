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

#define MT_MM 3
#define MT_NN 7
#define MT_WMASK 0xFFFFFFFFU
#define MT_UMASK 0xFFFFFFFEU
#define MT_LMASK 0x1U
#define MT_SHIFT0 12
#define MT_SHIFTB 7
#define MT_SHIFTC 15
#define MT_SHIFT1 18                        

constant unsigned int MT_NN_MINUS_ONE = MT_NN - 1;

kernel void bernoulli_kernel(device const mt_struct_stripped* d_MT,
                             device const int* n_rng_per_thread,
                             device const int* n_threads,
                             device unsigned char* result,
                             device unsigned int* MT_state,
                             uint index [[thread_position_in_grid]],
                             uint index_in_threadgroup [[thread_position_in_threadgroup]])
{
    //Create local states
    //unsigned int threshold = (unsigned int)(0.5 * 4294967295);
    const mt_struct_stripped mt_params = d_MT[index % 1024];

    // Allocate local state in private memory for faster access
    unsigned int mt[MT_NN];

    int iState;
    for(iState = 0; iState < MT_NN; iState++){
        mt[iState] = MT_state[(index) * (MT_NN + 1) + iState];
    }

    unsigned int mti1 = mt[0];
    unsigned int mtiM;
    unsigned int x;
    
    int iState1, iStateM;

    //Generate _nRngPerThread random numbers
    for (int iOut = 0, iState = 0; iOut < *n_rng_per_thread; iOut++) {
        iState1 = (iState + 1) & MT_NN_MINUS_ONE;
        iStateM = (iState + MT_MM) & MT_NN_MINUS_ONE;
        
        mtiM = mt[iStateM];

        // MT recurrence
        x = (mti1 & MT_UMASK) | (mt[iState1] & MT_LMASK);
        x = mtiM ^ (x >> 1) ^ ((x & 1U) ? mt_params.matrix_a : 0);

        mt[iState] = x;
        iState = iState1;
        mti1 = mt[iState1];

        // Tempering transformation
        x ^= (x >> MT_SHIFT0);
        x ^= (x << MT_SHIFTB) & mt_params.mask_b;
        x ^= (x << MT_SHIFTC) & mt_params.mask_c;
        x ^= (x >> MT_SHIFT1);

        //result[index + _nThreads * iOut] = (unsigned char) (x < threshold) ? 1 : 0;
        result[index + *n_threads * iOut] = (x >> 31) & 1U;
    }

    //Save current mt state
    //Unrolled loop
    MT_state[index * 8 + 0] = mt[0];
    MT_state[index * 8 + 1] = mt[1];
    MT_state[index * 8 + 2] = mt[2];
    MT_state[index * 8 + 3] = mt[3];
    MT_state[index * 8 + 4] = mt[4];
    MT_state[index * 8 + 5] = mt[5];
    MT_state[index * 8 + 6] = mt[6];

    // The last part of the original loop, setting the iState
    MT_state[index * 8 + 7] = iState;
}