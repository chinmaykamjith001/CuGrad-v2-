#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <algorithm> // For sorting/unique if you really want Set behavior
#include <cmath>

class Tensor{
public:
    std::vector<double>data;
    std::vector<int>shape;
    std::vector<int>strides;

    Tensor(std::vector<double> input_data, std::vector<int> input_shape){
        data = input_data;
        shape = input_shape;

        strides = {shape[1], 1};
    }

    double get_value(int row, int col){
        int flat = row * strides[0] + col * strides[1];
        return data[flat];
    }
};

int main() {
    std::cout << "--- Testing Tensor Indexing ---\n";
    
    // Create a 2x3 matrix
    // [ 10, 20, 30 ]
    // [ 40, 50, 60 ]
    std::vector<double> my_data = {10, 20, 30, 40, 50, 60};
    std::vector<int> my_shape = {2, 3};
    
    Tensor A(my_data, my_shape);
    
    // Let's try to get Row 1, Column 0 (Should be 40)
    // Note: Programmers start counting at 0!
    double val = A.get_value(1, 0);
    
    std::cout << "Expected: 40 | Your Output: " << val << "\n";

    return 0;
}
