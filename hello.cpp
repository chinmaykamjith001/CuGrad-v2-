#include <iostream>  // This lets us print to the screen

// 1. The Blueprint (The Class)
class Value {
public:            // "public" means everyone can see these variables
    float data;    // The actual number (like 2.0 or -3.0)
    float data1;
    

    // 2. The Constructor (Like __init__ in Python)
    // This runs automatically when you create a new Value.
    Value(float val) {
        data = val;
        data1 = val-1;
    }

    Value operator+(Value other){
        float new_data = data + other.data;
        Value out(new_data);
        return out;
    }

    // 3. A Method (Function inside a class)
    void print() {
        // std::cout is "print"
        // << acts like a funnel pushing text to the screen
        std::cout << "Value is: " << data <<  " and data1 is " << data1 << "\n";
    }
};

// 4. The Main Function (Where the program starts)
int main() {
    // Create an object 'a' from the Value blueprint
    // We pass 2.5 into the constructor
    Value a(2.5); 
    Value b(3);
    // Call the print method
    Value c = a.data + b.data;
    Value d = a + b;
    d.print();
    c.print();
    std::cout << d.data;
    return 0; // 0 means "Success"
}