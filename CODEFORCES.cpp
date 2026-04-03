#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <ctype.h>

int main(){
    std::string s;
    std::string t;
    std::cin >> s;
    std::cin >> t;
    int counter = 0;
    int size = t.length() - 1;

    for (int i = 0; i < s.length() / 2; i++) {
        char ss = s[i];
        char tt = t[size-i];
        if (ss == tt) {
            counter++;
            std::cout << ss << " " << tt;
        }
    }
    if (counter == s.length()) {
        std::cout << "YES";
    }
    else{std::cout << "NO";}


    return 0;
}