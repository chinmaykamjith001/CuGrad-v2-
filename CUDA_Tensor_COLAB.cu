%%writefile engine.cu

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
    // 1. Hardware Storage Primitives
    double* data;
    double* grad;
    int total_elements;

    // 2. Geometry Layout Tracking
    std::vector<int> shape;
    std::vector<int> strides;

    // 3. Graph Logic Primaries
    std::vector<std::shared_ptr<Tensor>> _prev;
    std::function<void()> _backward;

    // Constructor A (Initializer for inputs, weights, etc.)
    Tensor(const std::vector<double>& input_data, std::vector<int> input_shape, std::vector<std::shared_ptr<Tensor>> children = {}) {
        shape = input_shape;
        _prev = children;
        _backward = [](){};
        strides = calc_strides(shape);

        // Fixed: Mapped loop variable correctly to s
        total_elements = 1;
        for (int s : shape) total_elements *= s;

        // Fixed: Fixed typo on cudaMallocManaged for gradients
        cudaMallocManaged(&data, total_elements * sizeof(double));
        cudaMallocManaged(&grad, total_elements * sizeof(double));

        for(int i = 0; i < total_elements; i++){
            data[i] = input_data[i];
            grad[i] = 0.0;
        }
    }

    // Constructor B (Empty allocation tracker for operator outputs)
    Tensor(std::vector<int> input_shape, std::vector<std::shared_ptr<Tensor>> children = {}) {
        shape = input_shape;
        _prev = children;
        _backward = [](){};
        strides = calc_strides(shape);

        // Fixed: Mapped loop variable correctly to s
        total_elements = 1;
        for (int s : shape) total_elements *= s;

        // Fixed: Fixed typo on cudaMallocManaged for gradients
        cudaMallocManaged(&data, total_elements * sizeof(double));
        cudaMallocManaged(&grad, total_elements * sizeof(double));

        for(int i = 0; i < total_elements; i++){
            data[i] = 0.0;
            grad[i] = 0.0;
        }
    }

    // Hardware Destructor (RAII)
    ~Tensor() {
        cudaFree(data);
        cudaFree(grad);
    }

    double get_value(int row, int col) {
        int flat = row * strides[0] + col * strides[1];
        return data[flat];
    }

    // Topological Sort Backpropagation Execution
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

        // Fixed: Replaced vector range loop with index loop for raw pointers
        for (int i = 0; i < total_elements; ++i) {
            grad[i] = 1.0;
        }

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
    for (int i = 0; i < self->total_elements; i++) {
        total += self->data[i];
    }

    // Output is always a 1x1 shape
    auto out = std::make_shared<Tensor>(std::vector<double>{total}, std::vector<int>{1}, std::vector<std::shared_ptr<Tensor>>{self});

    out->_backward = [self, out]() {
        for (int i = 0; i < self->total_elements; i++) {
            self->grad[i] += 1.0 * out->grad[0];
        }
    };
    return out;
}

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> self) {
    double total = 0;
    for (int i = 0; i < self->total_elements; i++) total += self->data[i];
    double m = total / self->total_elements;

    auto out = std::make_shared<Tensor>(std::vector<double>{m}, std::vector<int>{1},
                                        std::vector<std::shared_ptr<Tensor>>{self});
    out->_backward = [self, out]() {
        double scale = 1.0 / self->total_elements;
        for (int i = 0; i < self->total_elements; i++) {
            self->grad[i] += scale * out->grad[0];
        }
    };
    return out;
}

__global__ void addForwardKernel(const double* A, const double* B, double* out, int a_rows, int a_cols, int b_rows, int b_cols, int out_rows, int out_cols, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i+=stride){
        int r = i / out_cols;
        int c = i % out_cols;

        int r_A = (a_rows == 1) ? 0 : r;
        int r_B = (b_rows == 1) ? 0 : r;

        int idx_A = r_A * a_cols + c;
        int idx_B = r_B * b_cols + c;

        out[i] = A[idx_A] + B[idx_B];

    }
}

__global__ void addBackwardKernel(double* A_grad, double* B_grad, const double* out_grad, int a_rows, int a_cols, int b_rows, int b_cols, int out_rows, int out_cols, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i+=stride){
        int r = i / out_cols;
        int c = i % out_cols;

        int r_A = (a_rows == 1) ? 0 : r;
        int r_B = (b_rows == 1) ? 0 : r;

        int idx_A = r_A * a_cols + c;
        int idx_B = r_B * b_cols + c;

        atomicAdd(&A_grad[idx_A], out_grad[i]);
        atomicAdd(&B_grad[idx_B], out_grad[i]);
    }
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

    auto out = std::make_shared<Tensor>(std::vector<int>{max_rows, max_cols}, std::vector<std::shared_ptr<Tensor>>{A, B});

    int size = out->total_elements;

    int blocksize = 256;
    int numblocks = (size + blocksize -1) / blocksize;

    // Forward pass with 2D broadcasting
    addForwardKernel<<<numblocks, blocksize>>>(A->data, B->data, out->data, A->shape[0], A->shape[1], B->shape[0], B->shape[1], out->shape[0], out->shape[1], size);
    cudaDeviceSynchronize();

    // Backward pass with Unbroadcasting accumulator
    out->_backward = [A, B, out, size, numblocks, blocksize](){
        addBackwardKernel<<<numblocks, blocksize>>>(A->grad, B->grad, out->grad, A->shape[0], A->shape[1], B->shape[0], B->shape[1], out->shape[0], out->shape[1], size);
        cudaDeviceSynchronize();
    };

    return out;
}

__global__ void subForwardKernel(const double* A, const double* B,  double* out, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i+=stride){
        out[i] = A[i] - B[i];
    }
}

__global__ void subBackwardKernel(const double* out_grad, double* A_grad, double* B_grad, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n; i += stride){
        A_grad[i] += out_grad[i];
        B_grad[i] += -1.0 * out_grad[i];
    }
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B){
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1]){
        throw std::runtime_error("Shape mismatch!");
    }
    auto out = std::make_shared<Tensor>(A->shape, std::vector<std::shared_ptr<Tensor>>{A, B});

    int size = out->total_elements;

    int blockSize = 256;
    int blockNum = (size + blockSize -1) / blockSize;

    subForwardKernel<<<blockNum, blockSize>>>(A->data, B->data, out->data, size);

    cudaDeviceSynchronize();

    out->_backward = [A, B, out, size, blockSize, blockNum](){
        subBackwardKernel<<<blockNum, blockSize>>>(out->grad, A->grad, B->grad, size);
        cudaDeviceSynchronize();
    };

    return out;
}

__global__ void mulForwardKernel(const double* A, const double* B, double* out, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = A[i] * B[i];
    }
}

__global__ void mulBackwardKernel(const double* out_grad, const double* A_data, const double* B_data, double* A_grad, double* B_grad, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i+= stride){
        A_grad[i] += out_grad[i] * B_data[i];
        B_grad[i] += out_grad[i] * A_data[i];
    }
}
std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B){
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1]){
        throw std::runtime_error("Not right size!");
    }

    auto out = std::make_shared<Tensor>(A->shape, std::vector<std::shared_ptr<Tensor>>{A, B});
    int size = out->total_elements;

    int blocksize = 256;
    int numblocks = (size + blocksize - 1) / blocksize;

    //forward
    mulForwardKernel<<<numblocks, blocksize>>>(A->data, B->data, out->data, size);
    cudaDeviceSynchronize();

    //backward
    out->_backward = [A, B, out, size, blocksize, numblocks](){
        mulBackwardKernel<<<numblocks, blocksize>>>(out->grad, A->data, B->data, A->grad, B->grad, size);
        cudaDeviceSynchronize();
    };

    return out;
}

__global__ void divForwardKernel(const double* A, const double* B, double* out, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = A[i] * (1.0/B[i]);
    }
}

__global__ void divBackwardKernel(const double* out_grad, double* a_grad, double* b_grad, const double* a_data, const double* b_data, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i += stride){
        a_grad[i] += out_grad[i] * (1.0/b_data[i]);
        b_grad[i] += out_grad[i] * ((-a_data[i]) / ((b_data[i] * b_data[i])));
    }
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B ){
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1]){
        throw std::runtime_error("Not right size!");
    }
    auto out = std::make_shared<Tensor>(A->shape, std::vector<std::shared_ptr<Tensor>>{A, B});
    int size = out->total_elements;

    int blocksize = 256;
    int numblocks = (size + blocksize - 1) / blocksize;

    //forward
    divForwardKernel<<<numblocks, blocksize>>>(A->data, B->data, out->data, size);
    cudaDeviceSynchronize();


    //backward

    out->_backward = [A, B, out, size, numblocks, blocksize](){
        divBackwardKernel<<<numblocks, blocksize>>>(out->grad, A->grad, B->grad, A->data, B->data, size);
        cudaDeviceSynchronize();
    };

    return out;
}


/////////////////////////////////////
// GPU Workhorses (2D Kernels)
/////////////////////////////////////

// 1. FORWARD KERNEL: Maps a 2D grid to compute individual dot products
__global__ void matmulForwardKernel(const double* A, const double* B, double* out,
                                    int M, int N, int K,
                                    int A_s0, int A_s1, int B_s0, int B_s1) {
    // Determine the unique 2D coordinate for this specific thread
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Maps to Columns (N)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Maps to Rows (M)

    // Hardware Boundary Guard: Blocks are 16x16. If your matrix size isn't a
    // perfect multiple of 16, threads on the outer edge must sit out idle!
    if (row < M && col < N) {
        double sum = 0.0;
        // Accumulate the dot product along the shared internal dimension K
        for (int k = 0; k < K; ++k) {
            sum += A[row * A_s0 + k * A_s1] * B[k * B_s0 + col * B_s1];
        }
        // Write the finished dot product to the flat output index
        out[row * N + col] = sum;
    }
}

// 2. BACKWARD KERNEL A: Computes A_grad = out_grad x B^T (Dimensions: M x K)
__global__ void matmulBackwardAKernel(const double* out_grad, const double* B, double* A_grad,
                                      int M, int N, int K, int B_s0, int B_s1) {
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x; // Maps to K
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // Maps to M

    if (row < M && k_idx < K) {
        double sum = 0.0;
        for (int col = 0; col < N; ++col) {
            // Strides are inverted to virtually evaluate B-Transposed
            sum += out_grad[row * N + col] * B[k_idx * B_s0 + col * B_s1];
        }
        atomicAdd(&A_grad[row * K + k_idx], sum);
    }
}

// 3. BACKWARD KERNEL B: Computes B_grad = A^T x out_grad (Dimensions: K x N)
__global__ void matmulBackwardBKernel(const double* out_grad, const double* A, double* B_grad,
                                      int M, int N, int K, int A_s0, int A_s1) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Maps to N
    int k_idx = blockIdx.y * blockDim.y + threadIdx.y; // Maps to K

    if (k_idx < K && col < N) {
        double sum = 0.0;
        for (int row = 0; row < M; ++row) {
            // Strides are inverted to virtually evaluate A-Transposed
            sum += A[row * A_s0 + k_idx * A_s1] * out_grad[row * N + col];
        }
        atomicAdd(&B_grad[k_idx * N + col], sum);
    }
}

/////////////////////////////////////
// C++ Operator Wrapper
/////////////////////////////////////
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B) {
    // 1. Matrix Shape Compatibility Verification
    if (A->shape[1] != B->shape[0]) {
        throw std::runtime_error("Matrix dimensions mismatch for matmul!");
    }

    int M = A->shape[0]; // Rows of A
    int K = A->shape[1]; // Inner dimension (Cols of A / Rows of B)
    int N = B->shape[1]; // Columns of B

    std::vector<int> out_shape{M, N};
    auto out = std::make_shared<Tensor>(out_shape, std::vector<std::shared_ptr<Tensor>>{A, B});

    // 2. Configure the 2D Spatial Hardware Layout
    // We create square blocks of 16x16 threads (256 threads total per block)
    dim3 threadsPerBlock(16, 16);

    // Calculate how many blocks are needed to cover the output's 2D grid dimensions
    dim3 numBlocksForward((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                          (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 3. Launch 2D Forward Kernel
    matmulForwardKernel<<<numBlocksForward, threadsPerBlock>>>(
        A->data, B->data, out->data, M, N, K,
        A->strides[0], A->strides[1], B->strides[0], B->strides[1]
    );
    cudaDeviceSynchronize();

    // 4. Register the 2D Backpropagation Graph Closure
    out->_backward = [A, B, out, M, N, K, threadsPerBlock]() {
        // A_grad matches A's dimensions (M x K)
        dim3 numBlocksA((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // B_grad matches B's dimensions (K x N)
        dim3 numBlocksB((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (K + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Fire both gradient calculations in parallel execution lanes
        matmulBackwardAKernel<<<numBlocksA, threadsPerBlock>>>(
            out->grad, B->data, A->grad, M, N, K, B->strides[0], B->strides[1]
        );
        matmulBackwardBKernel<<<numBlocksB, threadsPerBlock>>>(
            out->grad, A->data, B->grad, M, N, K, A->strides[0], A->strides[1]
        );
        cudaDeviceSynchronize();
    };

    return out;
}

/////////////////////////////////////
//Non-Linearity
/////////////////////////////////////
__global__ void reluForwardKernel(const double* A, double* out, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = (A[i] > 0) ? A[i] : 0.0;
    }
}

__global__ void reluBackwardKernel(double* A_grad, const double* A_data, const double* out_grad, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i += stride){
        double derivative = (A_data[i] > 0) ? 1.0 : 0.0;
        A_grad[i] += derivative * out_grad[i];
    }
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> A){

    auto out = std::make_shared<Tensor>(A->shape, std::vector<std::shared_ptr<Tensor>>{A});
    int size = out->total_elements;
    int blocksize = 256;
    int numblocks = (size + blocksize - 1) / blocksize;

    reluForwardKernel<<<numblocks, blocksize>>>(A->data, out->data, size);
    cudaDeviceSynchronize();

    out->_backward = [A, out, size, numblocks, blocksize](){
        reluBackwardKernel<<<numblocks, blocksize>>>(A->grad, A->data, out->grad, size);
        cudaDeviceSynchronize();
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
        static std::default_random_engine gen(42);
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
/////////////////////////////////////
// GPU Workhorses (Optimization Kernel)
/////////////////////////////////////

/////////////////////////////////////
// GPU Workhorses (Optimization Kernel)
/////////////////////////////////////

// 1. SGD UPDATE KERNEL: Performs weight = weight - (lr * gradient) in parallel
__global__ void sgdUpdateKernel(double* data, const double* grad, double lr, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        data[i] -= lr * grad[i];
    }
}

// 2. C++ Host Wrapper Function
void sgd_update(std::shared_ptr<Tensor> p, double lr) {
    int size = p->total_elements;
    int blockSize = 256;
    int blockNum = (size + blockSize - 1) / blockSize;

    // Launch the hardware update loop
    sgdUpdateKernel<<<blockNum, blockSize>>>(p->data, p->grad, lr, size);

    // Sync to make sure weights are completely updated before the next epoch starts
    cudaDeviceSynchronize();
}

/////////////////////////////////////
// Main Loop
/////////////////////////////////////
int main() {
    auto start = std::chrono::high_resolution_clock::now();

    int batch_size = 5000;
    int num_features = 100;

    // Generate the same synthetic data as engine.cpp
    std::vector<double> xs_data(batch_size * num_features);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < batch_size * num_features; ++i) xs_data[i] = dist(gen);
    auto x = std::make_shared<Tensor>(xs_data, std::vector<int>{batch_size, num_features});

    std::vector<double> ys_data(batch_size);
    for (int i = 0; i < batch_size; ++i)
        ys_data[i] = (xs_data[i * num_features] > 0.0) ? 1.0 : -1.0;
    auto y = std::make_shared<Tensor>(ys_data, std::vector<int>{batch_size, 1});

    MLP model(num_features, {64, 1}); // Same architecture

    for (int k = 0; k < 700; ++k) {

        // 1. FORWARD PASS
        auto ypred = model(x);

        // 2. LOSS CALCULATION (Mean Squared Error)
        auto diff = operator-(ypred, y);
        auto loss = mean(operator*(diff, diff));  // ← mean, not sum

        // 3. BACKWARD PASS (Reset Gradients + Backprop)
        for (auto p : model.parameters()) {
            // FIXED: Replaced vector iterators with raw pointer arithmetic bounds
            std::fill(p->grad, p->grad + p->total_elements, 0.0);
        }
        loss->backward();

        // 4. UPDATE (Gradient Descent - Now fully GPU parallelized!)
        double learning_rate = 0.01;
        for (auto p : model.parameters()) {
            sgd_update(p, learning_rate);
        }

        if (k % 50 == 0) {
            std::cout << "Epoch " << k << " | Loss: " << loss->data[0] << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}