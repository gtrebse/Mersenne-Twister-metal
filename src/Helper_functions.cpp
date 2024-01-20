#include "Helper_functions.hpp"
#include <cmath>
#include <iostream>

double calculateRMSE(const std::vector<double>& predictions) {
    double mse = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double error = predictions[i] - 0.5;
        mse += error * error;
    }
    mse /= static_cast<double>(predictions.size());
    return std::sqrt(mse);
}

double calculateMAE(const std::vector<double>& predictions) {
    double mae = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double error = std::abs(predictions[i] - 0.5);
        mae += error;
    }
    mae /= static_cast<double>(predictions.size());
    return mae;
}

std::vector<long> createRoundedExponentialVector(const double initial_value, const double final_value, const int number_of_elements) {
    // Calculate the factor
    const double factor = std::pow(final_value / initial_value, 1.0 / (number_of_elements - 1));

    // Create and fill the vector
    std::vector<long> exponential_vector;
    exponential_vector.reserve(number_of_elements);

    double current_value = initial_value;
    for (int i = 0; i < number_of_elements; ++i) {
        exponential_vector.push_back(std::round(current_value));
        current_value *= factor;
    }

    // Output the vector to check the values
    for (int i = 0; i < number_of_elements; ++i) {
        std::cout << "Element " << i << ": " << exponential_vector[i] << std::endl;
    }

    return exponential_vector;
}