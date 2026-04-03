#include <iostream>

int main() {
    // 1. Loop from 1 to 20
    for (int i = 1; i <= 20; i++) {
        
        // 2. Check for 15 FIRST (Divisible by both 3 and 5)
        // If we checked 3 first, 15 would just print "Fizz" and stop!
        if (i % 3 == 0 && i % 5 == 0) {
            std::cout << "FizzBuzz\n";
        }
        // 3. Check for 3
        else if (i % 3 == 0) {
            std::cout << "Fizz\n";
        }
        // 4. Check for 5
        else if (i % 5 == 0) {
            std::cout << "Buzz\n";
        }
        // 5. If none of the above, print the number
        else {
            std::cout << i << "\n";
        }
    }
    return 0;
}