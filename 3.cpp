#include <iostream>

void swap(int& a, int& b){
        int hold = a;
        a = b;
     
        b = hold;
    }

int main() {
    int a = 1;
    int b = 2;
    std::cout << "before" << a << " & " << b << "\n";

    swap(a, b);
    std::cout << "after" << a << " & " << b;
    return 0;
}

