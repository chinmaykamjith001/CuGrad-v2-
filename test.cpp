#include <iostream>
#include <vector> // We need this library for lists

class Value {
public:
    float data;
    // A list of "Pointers" (addresses) to other Values
    // <Value*> means "List of addresses where Values live"
    std::vector<Value*> children; 

    // Constructor
    Value(float val) {
        data = val;
    }

    // Addition with History
    Value operator+(Value& other) { // & means "Look at the original box, don't copy it"
        Value out(data + other.data);
        
        // Save the ADDRESS of the parents
        out.children.push_back(this);  // "this" is my own address
        out.children.push_back(&other); // "&" gets the address of 'other'
        
        return out;
    }
};

int main() {
    Value a(2.0);
    Value b(3.0);
    
    
    // We do this weird syntax just for this baby step to make the memory addresses stable
    // Don't worry about why just yet, just know it lets us track them.
    Value c = a + b;

    std::cout << "c.data = " << c.data << "\n";
    
    // Let's verify c remembers its parents!
    std::cout << "c has " << c.children.size() << " parents.\n";
    
    // We use '->' to access data through a pointer (address)
    std::cout << "Parent 1 data: " << c.children[0]->data << "\n";
    std::cout << "Parent 2 data: " << c.children[1]->data << "\n";
    
    return 0;
}