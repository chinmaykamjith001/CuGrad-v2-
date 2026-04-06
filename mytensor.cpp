#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <algorithm> // For sorting/unique if you really want Set behavior
#include <cmath>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<double>data;
    std::vector<int>shape;
    std::vector<int>strides;
    std::vector<double>grad;

    std::vector<std::shared_ptr<Tensor>> _prev;
    std::function<void()> _backward;

    Tensor(std::vector<double> input_data, std::vector<int> input_shape, std::vector<std::shared_ptr<Tensor>> children ={}){
        data = input_data;
        shape = input_shape;

        grad = std::vector<double>(data.size(), 0.0);

        _prev = children;
        _backward = [](){};

        strides = {shape[1], 1};
    }

    double get_value(int row, int col){
        int flat = row * strides[0] + col * strides[1];
        return data[flat];
    }

    std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> self) {
    double total = 0;
    for (double d : self->data) total += d;
    
    // Output is always a 1x1 shape
    auto out = std::make_shared<Tensor>(std::vector<double>{total}, std::vector<int>{1}, std::vector<std::shared_ptr<Tensor>>{self});
    
    out->_backward = [self, out]() {
        for (int i = 0; i < self->grad.size(); i++) {
            // The derivative is 1.0, so we just pass the 
            // incoming gradient (out->grad[0]) straight back.
            self->grad[i] += 1.0 * out->grad[0];
        }
    };
    return out;
    }

    //topological
    void backward() {
        std::vector<std::shared_ptr<Tensor>> topo;
        std::vector<std::shared_ptr<Tensor>> visited;

        std::function<void(std::shared_ptr<Tensor>)> build_topo = [&](std::shared_ptr<Tensor> v) {
            for (auto& x : visited) {
                if (x.get() == v.get()) return; 
            }
            visited.push_back(v);
            for (auto& child : v->_prev) build_topo(child);
            topo.push_back(v);
        };

        build_topo(shared_from_this());

        // Fill grad of final node with 1.0 (assuming output is scalar/loss)
        for (double& g : grad) g = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }
};


//------------------//
//Operators
//------------------//


std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B){
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1]){
        throw std::runtime_error("Shape mismatch!");
    }

    std::vector<double> out_data(A->data.size());

    for(int i = 0; i < A->data.size(); i++){
        out_data[i] = A->data[i] + B->data[i];
    }

    auto out = std::make_shared<Tensor>(out_data, A->shape, std::vector<std::shared_ptr<Tensor>>{A, B});

    out->_backward = [A, B, out](){
        for(int i = 0; i < out->grad.size(); i++){
            A->grad[i] += out->grad[i];
            B->grad[i] += out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B){
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1]){
        throw std::runtime_error("Shape mismatch!");
    }

    std::vector<double> out_data(A->data.size());

    for(int i = 0; i < A->data.size(); i++){
        out_data[i] = A->data[i] - B->data[i];
    }

    auto out = std::make_shared<Tensor>(out_data, A->shape, std::vector<std::shared_ptr<Tensor>>{A, B});

    out->_backward = [A, B, out](){
        for(int i = 0; i < out->grad.size(); i++){
            A->grad[i] += out->grad[i];
            B->grad[i] += -1 * out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B){
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1]){
        throw std::runtime_error("Not right size!");
    }

    //forward
    std::vector<double> out_data(A->data.size());
    for(int i = 0; i < A->data.size(); i++){
        out_data[i] = A->data[i] * B->data[i];
    }

    //backward
    auto out = std::make_shared<Tensor>(out_data, A->shape, std::vector<std::shared_ptr<Tensor>>{A, B});

    out->_backward = [A, B, out](){
        for(int i = 0; i < A->data.size(); i++){
            A->grad[i] += out->grad[i] * B->data[i]; 
            B->grad[i] += out->grad[i] * A->data[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B ){
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1]){
        throw std::runtime_error("Not right size!");
    }

    //forward
    std::vector<double> out_data(A->data.size());
    for(int i = 0; i < A->data.size(); i++){
        out_data[i] = A->data[i] * (1/B->data[i]);
    }

    //backward
    auto out = std::make_shared<Tensor>(out_data, A->shape, std::vector<std::shared_ptr<Tensor>>{A, B});

    out->_backward = [A, B, out](){
        for(int i = 0; i < A->data.size(); i++){
            double a = A->data[i];
            double b = B->data[i];
            
            // A's blame: out_grad * (1/b)
            A->grad[i] += out->grad[i] * (1.0 / b);
            
            // B's blame: out_grad * (-a / b^2)
            B->grad[i] += out->grad[i] * (-a / (b * b));
        }
    };

}


std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B){
    if (A->shape[1] != B->shape[0]){
        throw std::runtime_error("Not matching!");
    }

    int M = A->shape[0]; //rows of A
    int K = A->shape[1]; //same as rows of B
    int N = B->shape[1]; //cols of B

    std::vector<double> out_data(M*N, 0.0);
    std::vector<int> out_shape{M, N};

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            double sum = 0.0;
            for(int k = 0; k < K; k++){
                sum += A->data[i*A->strides[0] + k * A->strides[1]] * B->data[k * B->strides[0] + j * B->strides[1]];
            }
            out_data[i * N + j] = sum;
        }
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape, std::vector<std::shared_ptr<Tensor>>{A, B});

    out->_backward = [A,B,out,M,N,K](){
        for(int i = 0; i < M; i++){
            for(int k = 0; k < K; k++){
                double sum = 0.0;
                for(int j = 0; j < N; j++){
                    sum += out->grad[i * N + j] * B->data[k * N + j];
                }
                A->grad[i * K + k] += sum;
            }
        }
        for(int i = 0; i < K; i++){
            for(int k = 0; k < N; k++){
                double sum = 0.0;
                for(int j = 0; j < M; j++){
                    sum += out->grad[j*N + k] * A->data[j*K+i];
                }
                B->grad[i * N + k] += sum;
            }
        }
    
    };
    return out;
}

//NON-LINEARITY 
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> A){
    std::vector<double> out_data(A->data.size());

    for(int i = 0; i < A->data.size(); i++){
        out_data[i] = (A->data[i] > 0) ? A->data[i] : 0.0; 
    }

    auto out = std::make_shared<Tensor>(out_data, A->shape, std::vector<std::shared_ptr<Tensor>>{A});

    out->_backward = [A, out](){
        for(int i = 0; i < A->data.size(); i++){
            double derivative = (A->data[i] > 0) ? 1.0 : 0.0;
            A->grad[i] += derivative * out->grad[i];
        }
    };
}




