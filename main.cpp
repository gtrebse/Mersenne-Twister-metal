#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <fstream>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"
#include "Mersenne_twister.hpp"
#include "Helper_functions.hpp"

const double p = 0.5;
const int num_MC = 10;
const int num_vars = 1000*1e5;

int main() {
    //Create GPU code / arrays
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MetalMT *MT_1,*MT_2;
    std::vector<double> results;
    std::vector<double> results_MAE;
    std::vector<double> results_RMSE;
    //long long time;
    std::vector<int> time;
    std::vector<double> time_avg;
    std::vector<int> vars;

    MT_1 = new MetalMT(device);

    const double initial_value = 1;
    const double final_value = 1e8;
    const int number_of_elements = 100;

    std::vector<long> exponential_vector = createRoundedExponentialVector(initial_value, final_value, number_of_elements);

    //MT_1->setSeed(10);
    //Run MC simulation with different number of vars
    for (long iterations : exponential_vector) {
        //Empty the buffer vectors
        results.clear();
        time.clear();

        for (int j = 0; j < num_MC; ++j) {
            auto start_time = std::chrono::high_resolution_clock::now(); // Start time measurement
            MT_1->sendComputeCommand(iterations);

            auto end_time = std::chrono::high_resolution_clock::now(); // End time measurement 
            std::vector<unsigned char> result1 = MT_1->getResult();
            results.push_back(static_cast<double>(std::accumulate(result1.begin(), result1.end(), 0)) / (iterations));
  
            time.push_back(static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()));

            std::cout << "Time required was: " << static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) << " us" << std::endl;
            std::cout << "Running MC simulation with " << iterations << " iterations." << std::endl;
            std::cout << "Average: " << results[j] << std::endl;
            std::cout << "time required:" << time[j] << std::endl;
        }

        // Output the average
        results_MAE.push_back(calculateRMSE(results));
        results_RMSE.push_back(calculateMAE(results));
        time_avg.push_back(static_cast<double>(static_cast<double>(std::accumulate(time.begin(), time.end(), 0.0)) / static_cast<double>(num_MC)));
        vars.push_back(iterations);
    }

    //Print vector results_avg
    for (double elem : results_MAE) {
            std::cout << elem << " ";
        }
    std::cout << std::endl;
    for (double elem : time_avg) {
            std::cout << elem << " ";
        }
    std::cout << std::endl;
    for (int elem : vars) {
            std::cout << elem << " ";
        }
    std::cout << std::endl;


    // Open a file in write mode
    std::ofstream file("./data/Results.csv");

    // Check if the file is open
    if (file.is_open()) {
        // Iterate through the vectors and write to the file
        for (size_t i = 0; i < results_MAE.size(); ++i) {
            file << results_MAE[i] << "," << results_RMSE[i] << "," << time_avg[i] << "," << vars[i] << "\n";
        }

        // Close the file
        file.close();
        std::cout << "Data saved to vectors.csv" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
    
    return 0;
}