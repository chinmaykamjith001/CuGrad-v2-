#include <iostream>

int main(){
    int a = 5;

    int* b = new int(10);

    std::cout << a << "\n";
    std::cout << *b;
    
    delete b;
}