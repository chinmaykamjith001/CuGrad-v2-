#include <iostream>

class Value{
public:
    float data;

    Value(float val){
        data = val;
    }
};

int main() {
    Value v(3.14);
    std::cout << v.data;
}