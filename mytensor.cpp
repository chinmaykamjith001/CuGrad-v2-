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
};

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
        for(int i = 0; i < M; i++){
            for(int k = 0; k < K; k++){
                double sum = 0.0;
                for(int j = 0; j < N; j++){
                    sum += out->grad[i * N + j] * A->data[k * N + j];
                }
                B->grad[i * K + k] += sum;
            }
        }
    return out;
    };
}
    // The Engine: Topological Sort + Backprop
    void backward() {
        // 1. Prepare the lists
        std::vector<std::shared_ptr<Value>> topo;
        std::vector<std::shared_ptr<Value>> visited;

        // 2. Define the recursive helper (Lambda)
        std::function<void(std::shared_ptr<Value>)> build_topo = 
            [&](std::shared_ptr<Value> v) {
                
                // Check if already visited (Pointer comparison)
                for (auto& x : visited) {
                    if (x == v) return; 
                }
                visited.push_back(v);

                // Recursively visit all children (parents in the graph)
                for (auto& child : v->_prev) {
                    build_topo(child);
                }

                // Add self to the topological order (Post-order)
                topo.push_back(v);
            };

        // 3. Start the recursion from THIS node
        build_topo(shared_from_this());

        // 4. THE SPARK: Set the gradient of the final node to 1.0
        grad = 1.0;

        // 5. Run backward pass in REVERSE topological order
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

};

//addition
std::shared_ptr<Value> operator+(std::shared_ptr<Value> self, std::shared_ptr<Value> other){
    auto out = std::make_shared<Value>(
        self->data + other->data,
        std::vector<std::shared_ptr<Value>>{self, other},
        "+"
    );

    out->_backward = [self, other, out](){
        self->grad += 1 * out->grad;
        other->grad += 1 * out->grad;
    };

    return out;
}

std::shared_ptr<Value> operator+(std::shared_ptr<Value> self, double other){
    auto other_val = std::make_shared<Value>(other);
    return self + other_val;
}

std::shared_ptr<Value> operator+(double self_val, std::shared_ptr<Value> other){
    auto self = std::make_shared<Value>(self_val);
    return self + other;
}

//multiply
std::shared_ptr<Value> operator*(std::shared_ptr<Value> self, std::shared_ptr<Value> other){
    auto out = std::make_shared<Value>(
        self->data * other->data,
        std::vector<std::shared_ptr<Value>> {self, other},
        "*"
    );

    out->_backward = [self, other, out](){
        self->grad += other->data * out->grad;
        other->grad += self->data * out->grad;
    };

    return out;
}

std::shared_ptr<Value> operator*(std::shared_ptr<Value> self, double other){
    auto other_val = std::make_shared<Value>(other);
    return self * other_val;
}

std::shared_ptr<Value> operator*(double self, std::shared_ptr<Value> other){
    auto self_val = std::make_shared<Value>(self);
    return self_val * other;
}

//powers
std::shared_ptr<Value> pow(std::shared_ptr<Value> self, double exponent){
    double data = std::pow(self->data, exponent);
    auto out = std::make_shared<Value>(
        data,
        std::vector<std::shared_ptr<Value>> {self},
        "**" + std::to_string(exponent)
    );

    out->_backward = [self, exponent, out](){
        double derivative = exponent * std::pow(self->data, (exponent - 1));
        self->grad += derivative * out->grad;   
    };

    return out;
}

//division
std::shared_ptr<Value> operator/(std::shared_ptr<Value> self, std::shared_ptr<Value> other){
    return self * pow(other, -1);
}

std::shared_ptr<Value> operator/(std::shared_ptr<Value> self, double other){
    return self * (1/other);
}

std::shared_ptr<Value> operator/(double self, std::shared_ptr<Value> other){
    auto self_val = std::make_shared<Value>(self);
    return self_val * pow(other, -1);
}

//subtraction
std::shared_ptr<Value> operator-(std::shared_ptr<Value> self){
    return self * -1;
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> self, std::shared_ptr<Value> other){
    return self + (-other);
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> self, double other){
    auto other_val = std::make_shared<Value>(other);
    return self + (-other_val); //prevent recursion by not doing self - other
}

std::shared_ptr<Value> operator-(double self, std::shared_ptr<Value> other){
    auto self_val = std::make_shared<Value>(self);
    return self_val + (-other);
}

//exp
std::shared_ptr<Value> exp(std::shared_ptr<Value> self){
    double data = std::exp(self->data);

    auto out = std::make_shared<Value> (
        data,
        std::vector<std::shared_ptr<Value>> {self},
        "exp"
    );

    out->_backward = [self, out](){
        self->grad += out->data * out->grad;
    };

    return out;
}

//tanh
std::shared_ptr<Value> tanh(std::shared_ptr<Value> self){
    double x = self->data;
    double data = std::tanh(x);

    auto out = std::make_shared<Value>(
        data,
        std::vector<std::shared_ptr<Value>> {self},
        "tanh"
    );

    out->_backward = [self, data, out](){
        self->grad += (1.0 - (data * data)) * out->grad; 
    };

    return out;
}

//relu
std::shared_ptr<Value> relu(std::shared_ptr<Value> self){
    //forward compute
    double data = self->data > 0 ? self->data : 0;

    auto out = std::make_shared<Value>(
        data,
        std::vector<std::shared_ptr<Value>> {self},
        "relu"
    );

    out->_backward = [self, out](){  
        self->grad += (out->data > 0 ? 1 : 0) * out->grad;
    };

    return out;
}

#include <random>

class Neuron {
public:
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    bool nonlin;

    Neuron(int nin, bool nonlin=true) {
        this->nonlin = nonlin;
        // 1. Setup random number generator for weights [-1, 1]
        static std::default_random_engine gen;
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        // 2. Initialize weights
        for (int i = 0; i < nin; ++i) {
            w.push_back(std::make_shared<Value>(dist(gen)));
        }
        // 3. Initialize bias
        b = std::make_shared<Value>(dist(gen));
    }

    // The forward pass: y = relu(sum(wi*xi) + b)
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x) {
        auto act = b; 
        for (size_t i = 0; i < w.size(); ++i) {
            act = act + (w[i] * x[i]); // Summing inputs * weights [cite: 170-172, 262]
        }
        return nonlin ? tanh(act) : act;
    }

    std::vector<std::shared_ptr<Value>> parameters() {
        std::vector<std::shared_ptr<Value>> params = w;
        params.push_back(b);
        return params;
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;

    // nin: number of inputs; nout: number of neurons in this layer
    Layer(int nin, int nout, bool nonlin=true) {
        for (int i = 0; i < nout; ++i) {
            neurons.emplace_back(nin, nonlin); // 
        }
    }

    // Forward pass: returns the output of every neuron in the layer
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x) {
        std::vector<std::shared_ptr<Value>> outs;
        for (auto& n : neurons) {
            outs.push_back(n(x)); // 
        }
        return outs;
    }

    // Flattens all parameters (w and b) from every neuron into one list
    std::vector<std::shared_ptr<Value>> parameters() {
        std::vector<std::shared_ptr<Value>> params;
        for (auto& n : neurons) {
            std::vector<std::shared_ptr<Value>> p = n.parameters(); // 
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};

class MLP{
public: 
    std::vector<Layer> layers;

    MLP(int nin, std::vector<int> nouts){
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());

        for (size_t i = 0; i < nouts.size(); ++i) {
            layers.emplace_back(sz[i], sz[i+1], i != nouts.size() - 1);
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x){
        for (auto& layer : layers){
            x = layer(x);
        }
        return x;
    }

    // Gathers every single weight and bias in the entire network
    std::vector<std::shared_ptr<Value>> parameters() {
        std::vector<std::shared_ptr<Value>> params;
        for (auto& layer : layers) {
            std::vector<std::shared_ptr<Value>> lp = layer.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        return params;
    }
};

int main() {

    std::vector<std::vector<std::shared_ptr<Value>>> xs = {
        { std::make_shared<Value>(2.0), std::make_shared<Value>(3.0), std::make_shared<Value>(-1.0) },
        { std::make_shared<Value>(3.0), std::make_shared<Value>(1.0), std::make_shared<Value>(0.5) },
        { std::make_shared<Value>(0.5), std::make_shared<Value>(1.0), std::make_shared<Value>(1.0) },
        { std::make_shared<Value>(1.0), std::make_shared<Value>(1.0), std::make_shared<Value>(1.0) }
    };

    std::vector<double> ys = {1, -1, -1, 1};

    MLP model(3, {3,3,3,3,1});

    for (int k = 0; k < 50000; ++k) {
        
        // --- FORWARD PASS ---
        std::vector<std::shared_ptr<Value>> ypred;
        for (auto& x : xs) {
            ypred.push_back(model(x)[0]); // Get the single output from the MLP [cite: 309]
        }

        // --- LOSS CALCULATION (Mean Squared Error) ---
        auto loss = std::make_shared<Value>(0.0);
        for (size_t i = 0; i < ys.size(); ++i) {
            auto diff = ypred[i] - ys[i];
            loss = loss + (diff * diff); // 
        }

        // --- BACKWARD PASS ---
        // Crucial: Reset gradients to zero before backprop 
        for (auto p : model.parameters()) {
            p->grad = 0.0;
        }
        
        loss->backward(); // [cite: 316]

        // --- UPDATE (Gradient Descent) ---
        double learning_rate = 0.001; // [cite: 319]
        for (auto p : model.parameters()) {
            p->data += -learning_rate * p->grad; // [cite: 319]
        }

        std::cout << "Epoch " << k << " | Loss: " << loss->data << std::endl; // [cite: 320]
    }

    return 0;
};