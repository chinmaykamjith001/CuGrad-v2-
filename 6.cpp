#include <iostream>

double add(double x, double y){
    return x + y;
}


int main(){
    double a;
    double b;
    char op;

    std::cout << "First number";
    std::cin >> a;
    std::cout << "second number";
    std::cin >> b;
    std::cout << "Operation";
    std::cin >> op;

    if (op == '+'){
        add(a,b);
     }
    else if (op == '-'){
        double c = a - b;
        std::cout << c;
    }
    else if (op == '*'){
        double c = a * b;
        std::cout << c;
    }
    else if (op == '/'){
        double c = a / b;
        std::cout << c;
    }
    else {
        std::cout << "Enter valid operation symbol";
    }
}