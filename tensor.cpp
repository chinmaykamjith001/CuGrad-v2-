#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

// ---------------------------------------------------------
// BROADCAST AND SHAPE UTILITIES
// ---------------------------------------------------------

std::vector<int> calc_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    }
    return strides;
}

int calc_size(const std::vector<int>& shape) {
    if (shape.empty()) return 1;
    int sz = 1;
    for (int s : shape) sz *= s;
    return sz;
}

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

std::vector<int> broadcast_strides(const std::vector<int>& b_shape, const std::vector<int>& orig_shape, const std::vector<int>& orig_strides) {
    std::vector<int> b_strides;
    int rank_diff = b_shape.size() - orig_shape.size();
    for (int i = 0; i < (int)b_shape.size(); ++i) {
        if (i < rank_diff) {
            b_strides.push_back(0); // Padded dimension
        } else {
            int orig_idx = i - rank_diff;
            if (orig_shape[orig_idx] == 1 && b_shape[i] > 1) {
                b_strides.push_back(0); // Broadcasted dimension 
            } else {
                b_strides.push_back(orig_strides[orig_idx]);
            }
        }
    }
    return b_strides;
}

std::vector<int> flat_to_nd(int flat_index, const std::vector<int>& shape) {
    std::vector<int> nd(shape.size());
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        nd[i] = flat_index % shape[i];
        flat_index /= shape[i];
    }
    return nd;
}

int nd_to_flat(const std::vector<int>& nd, const std::vector<int>& strides) {
    int flat = 0;
    for (size_t i = 0; i < nd.size(); ++i) {
        flat += nd[i] * strides[i];
    }
    return flat;
}

std::vector<double> unbroadcast_grad(const std::vector<double>& b_grad, const std::vector<int>& b_shape, const std::vector<int>& orig_shape) {
    if (b_shape == orig_shape) return b_grad; // Fast path
    
    std::vector<double> out_grad(calc_size(orig_shape), 0.0);
    std::vector<int> orig_strides = calc_strides(orig_shape);
    std::vector<int> b_strides_for_orig = broadcast_strides(b_shape, orig_shape, orig_strides);
    
    for (size_t i = 0; i < b_grad.size(); ++i) {
        std::vector<int> nd = flat_to_nd(i, b_shape);
        int out_flat = nd_to_flat(nd, b_strides_for_orig);
        out_grad[out_flat] += b_grad[i];
    }
    return out_grad;
}

// ---------------------------------------------------------
// TENSOR CLASS
// ---------------------------------------------------------

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<int> shape;
    std::vector<int> strides;
    
    std::function<void()> _backward;
    std::vector<std::shared_ptr<Tensor>> _prev;
    std::string _op;

    Tensor(std::vector<double> data, std::vector<int> shape, std::vector<std::shared_ptr<Tensor>> children = {}, std::string op = "") 
        : data(std::move(data)), shape(std::move(shape)), _prev(std::move(children)), _op(std::move(op)) {
        
        int sz = calc_size(this->shape);
        if (this->data.size() != sz) {
            if (this->data.size() == 1) { // autofill scalar
                this->data = std::vector<double>(sz, this->data[0]);
            } else {
                throw std::runtime_error("Data size does not match shape.");
            }
        }
        
        grad.resize(sz, 0.0);
        strides = calc_strides(this->shape);
        _backward = [](){};
    }
    
    Tensor(double val) : Tensor(std::vector<double>{val}, {1}) {}

    // Topological Sort + Backprop engine
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

// ---------------------------------------------------------
// TENSOR OPERATIONS
// ---------------------------------------------------------

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> self) {
    double total = 0;
    for (double d : self->data) total += d;
    
    auto out = std::make_shared<Tensor>(
        std::vector<double>{total}, 
        std::vector<int>{1}, 
        std::vector<std::shared_ptr<Tensor>>{self}, 
        "sum"
    );
    
    out->_backward = [self, out]() {
        for (size_t i = 0; i < self->grad.size(); ++i) {
            self->grad[i] += out->grad[0];
        }
    };
    return out;
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int size = calc_size(out_shape);
    std::vector<double> out_data(size);
    
    std::vector<int> a_b_strides = broadcast_strides(out_shape, a->shape, a->strides);
    std::vector<int> b_b_strides = broadcast_strides(out_shape, b->shape, b->strides);
    
    for (int i = 0; i < size; ++i) {
        std::vector<int> nd = flat_to_nd(i, out_shape);
        int a_idx = nd_to_flat(nd, a_b_strides);
        int b_idx = nd_to_flat(nd, b_b_strides);
        out_data[i] = a->data[a_idx] + b->data[b_idx];
    }
    
    auto out = std::make_shared<Tensor>(
        out_data, out_shape, std::vector<std::shared_ptr<Tensor>>{a, b}, "+"
    );
    
    out->_backward = [a, b, out]() {
        std::vector<double> a_grad = unbroadcast_grad(out->grad, out->shape, a->shape);
        std::vector<double> b_grad = unbroadcast_grad(out->grad, out->shape, b->shape);
        for(size_t i=0; i<a->grad.size(); ++i) a->grad[i] += a_grad[i];
        for(size_t i=0; i<b->grad.size(); ++i) b->grad[i] += b_grad[i];
    };
    return out;
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::vector<int> out_shape = broadcast_shape(a->shape, b->shape);
    int size = calc_size(out_shape);
    std::vector<double> out_data(size);
    
    std::vector<int> a_b_strides = broadcast_strides(out_shape, a->shape, a->strides);
    std::vector<int> b_b_strides = broadcast_strides(out_shape, b->shape, b->strides);
    
    for (int i = 0; i < size; ++i) {
        std::vector<int> nd = flat_to_nd(i, out_shape);
        int a_idx = nd_to_flat(nd, a_b_strides);
        int b_idx = nd_to_flat(nd, b_b_strides);
        out_data[i] = a->data[a_idx] * b->data[b_idx];
    }
    
    auto out = std::make_shared<Tensor>(
        out_data, out_shape, std::vector<std::shared_ptr<Tensor>>{a, b}, "*"
    );
    
    out->_backward = [a, b, out, a_b_strides, b_b_strides]() {
        std::vector<double> a_grad_b(out->grad.size());
        std::vector<double> b_grad_b(out->grad.size());
        for (int i = 0; i < (int)out->grad.size(); ++i) {
            std::vector<int> nd = flat_to_nd(i, out->shape);
            int a_idx = nd_to_flat(nd, a_b_strides);
            int b_idx = nd_to_flat(nd, b_b_strides);
            a_grad_b[i] = out->grad[i] * b->data[b_idx];
            b_grad_b[i] = out->grad[i] * a->data[a_idx];
        }
        std::vector<double> a_grad = unbroadcast_grad(a_grad_b, out->shape, a->shape);
        std::vector<double> b_grad = unbroadcast_grad(b_grad_b, out->shape, b->shape);
        for(size_t i=0; i<a->grad.size(); ++i) a->grad[i] += a_grad[i];
        for(size_t i=0; i<b->grad.size(); ++i) b->grad[i] += b_grad[i];
    };
    return out;
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, double val) {
    std::vector<double> out_data(a->data.size());
    for (size_t i = 0; i < a->data.size(); ++i) out_data[i] = a->data[i] * val;
    auto out = std::make_shared<Tensor>(out_data, a->shape, std::vector<std::shared_ptr<Tensor>>{a}, "*");
    out->_backward = [a, out, val]() {
        for(size_t i=0; i<a->grad.size(); ++i) a->grad[i] += out->grad[i] * val;
    };
    return out;
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, double val) {
    auto b = std::make_shared<Tensor>(val);
    return operator+(a, b);
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    return operator+(a, operator*(b, -1.0));
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape.size() != 2 || b->shape.size() != 2 || a->shape[1] != b->shape[0]) {
        throw std::runtime_error("Matmul shape error (requires 2D matching inner dims).");
    }
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];
    std::vector<double> out_data(M * N, 0.0);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0;
            for (int k = 0; k < K; ++k) {
                s += a->data[i * K + k] * b->data[k * N + j];
            }
            out_data[i * N + j] = s;
        }
    }
    
    auto out = std::make_shared<Tensor>(
        out_data, std::vector<int>{M, N}, std::vector<std::shared_ptr<Tensor>>{a, b}, "matmul"
    );
    
    out->_backward = [a, b, out, M, K, N]() {
        // dL/dA = C.grad @ B^T
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                double s = 0;
                for (int j = 0; j < N; ++j) {
                    s += out->grad[i * N + j] * b->data[k * N + j];
                }
                a->grad[i * K + k] += s;
            }
        }
        // dL/dB = A^T @ C.grad
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                double s = 0;
                for (int i = 0; i < M; ++i) {
                    s += a->data[i * K + k] * out->grad[i * N + j];
                }
                b->grad[k * N + j] += s;
            }
        }
    };
    return out;
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> self) {
    std::vector<double> out_data(self->data.size());
    for(size_t i=0; i<self->data.size(); ++i) {
        out_data[i] = self->data[i] > 0 ? self->data[i] : 0;
    }
    auto out = std::make_shared<Tensor>(
        out_data, self->shape, std::vector<std::shared_ptr<Tensor>>{self}, "relu"
    );
    out->_backward = [self, out]() {
        for(size_t i=0; i<self->grad.size(); ++i) {
            self->grad[i] += (out->data[i] > 0 ? 1.0 : 0.0) * out->grad[i];
        }
    };
    return out;
}

std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> self) {
    std::vector<double> out_data(self->data.size());
    for(size_t i=0; i<self->data.size(); ++i) {
        out_data[i] = std::tanh(self->data[i]);
    }
    auto out = std::make_shared<Tensor>(
        out_data, self->shape, std::vector<std::shared_ptr<Tensor>>{self}, "tanh"
    );
    out->_backward = [self, out]() {
        for(size_t i=0; i<self->grad.size(); ++i) {
            self->grad[i] += (1.0 - out->data[i] * out->data[i]) * out->grad[i];
        }
    };
    return out;
}

// ---------------------------------------------------------
// NEURAL NETWORK MODULES
// ---------------------------------------------------------

class Layer {
public:
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> b;
    bool nonlin;

    Layer(int nin, int nout, bool nonlin=true) {
        this->nonlin = nonlin;
        static std::default_random_engine gen(1337); 
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        std::vector<double> w_data(nin * nout);
        for(auto& d : w_data) d = dist(gen);
        w = std::make_shared<Tensor>(w_data, std::vector<int>{nin, nout});
        
        std::vector<double> b_data(nout);
        for(auto& d : b_data) d = dist(gen);
        b = std::make_shared<Tensor>(b_data, std::vector<int>{1, nout}); // [1, nout] enables broadcasting!
    }

    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> x) {
        auto out = operator+(matmul(x, w), b); 
        return nonlin ? tanh(out) : out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() {
        return {w, b};
    }
};

class MLP {
public: 
    std::vector<Layer> layers;

    MLP(int nin, std::vector<int> nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());

        for (size_t i = 0; i < nouts.size(); ++i) {
            layers.emplace_back(sz[i], sz[i+1], i != nouts.size() - 1);
        }
    }

    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> x) {
        for (auto& layer : layers){
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

// ---------------------------------------------------------
// TRAINING LOOP
// ---------------------------------------------------------

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

    MLP model(3, {4, 4, 1});

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
        double learning_rate = 0.05; 
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