#include <iostream>

int main() {
    int a = 10;
    
    // 'p' is a POINTER. It holds the ADDRESS of 'a', not the value 10.
    int* p = &a; 
    
    // We change the value AT that address to 20.
    *p = 30;
    
    int b = 10;
    b = 20; // We change b directly.

    std::cout << "Address of a: " << p << "\n";
    std::cout << "Value of a: " << a << "\n";
    std::cout << "Value of b: " << *p << "\n";
    
    return 0;
}