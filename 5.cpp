#include <iostream>
#include <vector>

int main(){
    std::vector<int> x;
    x.push_back(10);
    x.push_back(20);
    x.push_back(30);

    std::cout << x[0] << "\n";
    std::cout << x.size();
    

}