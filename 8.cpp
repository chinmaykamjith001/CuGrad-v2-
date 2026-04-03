#include <iostream>
#include <memory>
#include <vector>

class Node: public std::enable_shared_from_this<Node>{
public:
    int id;

    std::vector<std::shared_ptr<Node>> children;

    void addChild(std::shared_ptr<Node> child){
        children.push_back(child);
    }
}