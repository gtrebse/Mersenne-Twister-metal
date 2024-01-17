#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>

double calculateRMSE(const std::vector<double>& predictions);
double calculateMAE(const std::vector<double>& predictions);
std::vector<int> createRoundedExponentialVector(const double initial_value, const double final_value, const int number_of_elements); // Added semicolon here


#endif // HELPER_FUNCTIONS_H