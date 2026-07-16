#include <iostream>

void add(int x, int top) {
    if (x < top) {
        std::cout << x << std::endl;
        add(++x, top);
        
    } else {
        std::cout << "Done" << std::endl;
    }
}

int main() {
    add(0, 1000);
    return 0;
}
