#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
/////////////////////////////////////
//Tensor class
/////////////////////////////////////
std::vector<int> calc_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    }
    return strides;
}

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

        strides = calc_strides(shape);
    }

    double get_value(int row, int col){
        int flat = row * strides[0] + col * strides[1];
        return data[flat];
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


/////////////////////////////////////
//Operators
/////////////////////////////////////

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

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B){
    // Determine the broadcasted output shape
    int max_rows = std::max(A->shape[0], B->shape[0]);
    int max_cols = std::max(A->shape[1], B->shape[1]);
    
    // Safety check: ensure columns match, and rows match or one is a singleton (1)
    if (A->shape[1] != B->shape[1] || 
       (A->shape[0] != B->shape[0] && A->shape[0] != 1 && B->shape[0] != 1)) {
        throw std::runtime_error("Shape mismatch in operator+!");
    }

    std::vector<int> out_shape = {max_rows, max_cols};
    std::vector<double> out_data(max_rows * max_cols);

    // Forward pass with 2D broadcasting
    for (int r = 0; r < max_rows; ++r) {
        for (int c = 0; c < max_cols; ++c) {
            // If a tensor only has 1 row, stay at row 0 (broadcast it!)
            int r_A = (A->shape[0] == 1) ? 0 : r;
            int r_B = (B->shape[0] == 1) ? 0 : r;
            
            int idx_out = r * max_cols + c;
            int idx_A = r_A * A->shape[1] + c;
            int idx_B = r_B * B->shape[1] + c;
            
            out_data[idx_out] = A->data[idx_A] + B->data[idx_B];
        }
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape, std::vector<std::shared_ptr<Tensor>>{A, B});

    // Backward pass with Unbroadcasting accumulator
    out->_backward = [A, B, out, max_rows, max_cols](){
        for (int r = 0; r < max_rows; ++r) {
            for (int c = 0; c < max_cols; ++c) {
                int idx_out = r * max_cols + c;
                
                int r_A = (A->shape[0] == 1) ? 0 : r;
                int r_B = (B->shape[0] == 1) ? 0 : r;
                
                int idx_A = r_A * A->shape[1] + c;
                int idx_B = r_B * B->shape[1] + c;
                
                // Accumulate gradients back
                A->grad[idx_A] += out->grad[idx_out];
                B->grad[idx_B] += out->grad[idx_out];
            }
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
        for(size_t i = 0; i < out->grad.size(); i++){
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
    for(size_t i = 0; i < A->data.size(); i++){
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

    return out;

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

/////////////////////////////////////
//Non-Linearity
/////////////////////////////////////
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

    return out;
}

/////////////////////////////////////
//Broadcasting utils
/////////////////////////////////////



std::vector<int> broadcast_shape(const std::vector<int>& s1, const std::vector<int>& s2) {
    std::vector<int> out_shape;
    int rank = std::max(s1.size(), s2.size());
    for (int i = 0; i < rank; ++i) {
        int d1 = (i < rank - (int)s1.size()) ? 1 : s1[i - (rank - s1.size())];
        int d2 = (i < rank - (int)s2.size()) ? 1 : s2[i - (rank - s2.size())];
        if (d1 != d2 && d1 != 1 && d2 != 1) {
            throw std::runtime_error("Shapes are not broadcastable");
        }
        out_shape.push_back(std::max(d1, d2));
    }
    return out_shape;
}

std::vector<int> get_broadcast_strides(const std::vector<int>& in_shape, const std::vector<int> in_strides, const std::vector<int> out_shape){
    int in_rank = (int)in_shape.size();
    int out_rank = (int)out_shape.size();

    std::vector<int> b_strides(out_shape.size(), 0);

    for(int i = 0; i < in_rank; i ++){
        int out_idx = out_rank - 1 - i;
        int in_idx = in_rank - 1 - i;

        if(in_shape[in_idx] == out_shape[out_idx]){
            b_strides[out_idx] = in_strides[in_idx];
        }
    }
    return b_strides;
} 

///////////////////////////////////
//Neural Net Modules
///////////////////////////////////

class Layer {
public:
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> b;
    bool nonlin;

    Layer(int nin, int nout, bool nonlin = true){
        this->nonlin = nonlin;
        static std::default_random_engine gen(100);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);

        std::vector<double> w_data(nin * nout);
        for(auto& d : w_data) d = dist(gen);
        w = std::make_shared<Tensor>(w_data, std::vector<int>{nin, nout});

        std::vector<double> b_data(nout);
        for(auto& d : b_data) d = dist(gen);
        b = std::make_shared<Tensor>(b_data, std::vector<int>{1, nout});
    }

    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> x){
        auto out = operator+(matmul(x,w), b);
        return nonlin ? relu(out) : out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters(){
        return {w,b};
    }
};

class MLP{
public: 
    std::vector<Layer> layers;

    MLP(int nin, std::vector<int> nout){
        std::vector<int> hold = {nin};
        hold.insert(hold.end(), nout.begin(), nout.end());

        for (size_t i = 0; i < nout.size(); ++i) {
            layers.emplace_back(hold[i], hold[i+1], i != nout.size() - 1);
        }
    }

    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> x){
        for(auto& layer : layers){
            x = layer(x);
        }
        return x;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() {
        std::vector<std::shared_ptr<Tensor>> params;
        for (auto& layer : layers) {
            auto lp = layer.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        return params;
    }

};

///////////////////////////////////
//Main loop
///////////////////////////////////

int main() {

    auto start = std::chrono::high_resolution_clock::now();
    // Input: shape [4, 3] (batch size 4, features 3)
    std::vector<double> xs_data = {
        2.0, 3.0, -1.0,
        3.0, 1.0, 0.5,
        0.5, 1.0, 1.0,
        1.0, 1.0, 5.0,
    };
    auto x = std::make_shared<Tensor>(xs_data, std::vector<int>{4, 3});

    // Targets: shape [4, 1] 
    std::vector<double> ys_data = {1, -1, -1, 1};
    auto y = std::make_shared<Tensor>(ys_data, std::vector<int>{4, 1});

    MLP model(3, {128, 64, 32, 16, 8, 4, 1});

    for (int k = 0; k < 500; ++k) { // Tensor operations converge very efficiently or may require different epochs
        
        // 1. FORWARD PASS
        auto ypred = model(x);

        // 2. LOSS CALCULATION (Mean Squared Error)
        auto diff = operator-(ypred, y);
        auto loss = sum(operator*(diff, diff));

        // 3. BACKWARD PASS (Reset Gradients + Backprop)
        for (auto p : model.parameters()) {
            std::fill(p->grad.begin(), p->grad.end(), 0.0);
        }
        loss->backward();

        // 4. UPDATE (Gradient Descent)
        double learning_rate = 0.005; 
        for (auto p : model.parameters()) {
            for (size_t i = 0; i < p->data.size(); ++i) {
                p->data[i] -= learning_rate * p->grad[i];
            }
        }

        if (k % 50 == 0) {
            std::cout << "Epoch " << k << " | Loss: " << loss->data[0] << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Final demonstration of the forward pass outputs
    std::cout << "Predictions after training:" << std::endl;
    auto ypred = model(x);
    for (int i = 0; i < 4; ++i) {
        std::cout << "Pred: " << ypred->data[i] << " | Expected: " << ys_data[i] << std::endl;
    }
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
    

    return 0;
}