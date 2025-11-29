#include <iostream>
#include <vector>     // Still needed for training data std::vector<FixedVector<...>>
#include <array>      // For fixed-size vectors and internal matrix storage
#include <cmath>      // For std::exp, std::sqrt, std::tanh
#include <random>     // For std::random_device, std::mt19937, std::normal_distribution
#include <algorithm>  // For std::generate, std::fill
#include <iomanip>    // For std::fixed, std::setprecision

// --- Fixed-size data structures ---

// Type alias for a fixed-size vector (e.g., a layer's activations or biases)
template <size_t N>
using FixedVector = std::array<double, N>;

// Custom FixedMatrix struct for optimal memory locality and performance.
// Stores data in a flat std::array<double> in row-major order.
// Dimensions are known at compile-time, eliminating dynamic allocation and resizing overhead.
template <size_t Rows, size_t Cols>
struct FixedMatrix {
    std::array<double, Rows * Cols> data;

    // Default constructor value-initializes all elements to 0.0 (via std::array's aggregate initialization).
    FixedMatrix() noexcept = default;

    // Constructor to initialize all elements with a specific value.
    explicit FixedMatrix(double initial_val) noexcept {
        std::fill(data.begin(), data.end(), initial_val);
    }

    // Access operator for mutable elements. No bounds checking for performance.
    // Bounds are guaranteed by compile-time dimensions and careful usage.
    double& operator()(size_t r, size_t c) noexcept {
        return data[r * Cols + c];
    }

    // Access operator for const elements. No bounds checking for performance.
    const double& operator()(size_t r, size_t c) const noexcept {
        return data[r * Cols + c];
    }

    // Fills all existing elements to a specific value.
    void fill_all(double initial_value = 0.0) noexcept {
        std::fill(data.begin(), data.end(), initial_value);
    }

    // Check if the matrix is empty (compile-time constant for FixedMatrix).
    // For R, C > 0, this is always false.
    [[nodiscard]] constexpr bool empty() const noexcept {
        return Rows == 0 || Cols == 0;
    }
};

// --- Activation Functions ---

// f(x) = tanh(x)
// Operates in-place on 'result'. Template ensures size matching at compile time.
template <size_t N>
void tanh_activation_in_place(const FixedVector<N>& v, FixedVector<N>& result) noexcept {
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::tanh(v[i]);
    }
}

// f(x) = 1 / (1 + e^-x)
// Operates in-place on 'result'. Template ensures size matching at compile time.
template <size_t N>
void sigmoid_in_place(const FixedVector<N>& v, FixedVector<N>& result) noexcept {
    for (size_t i = 0; i < N; ++i) {
        result[i] = 1.0 / (1.0 + std::exp(-v[i]));
    }
}

// --- Matrix/Vector Operations (Optimized for Fixed-Size) ---

// Dot product/Matrix multiplication: V * M (1xN * NxM -> 1xM)
// result[j] = sum_k (v[k] * m[k][j])
template <size_t N_V, size_t M_ROWS, size_t M_COLS>
void dot_in_place(const FixedVector<N_V>& v, const FixedMatrix<M_ROWS, M_COLS>& m, FixedVector<M_COLS>& result) noexcept {
    static_assert(N_V == M_ROWS, "Dot product dimension mismatch (Vector * Matrix): Vector size != Matrix rows");
    
    result.fill(0.0); // Initialize result elements to zero
    
    // Modern compilers (like GCC/Clang) are very adept at auto-vectorizing
    // these kinds of fixed-size nested loops, potentially generating SSE/AVX instructions.
    // Loop order chosen for cache efficiency in row-major matrix `m`.
    for (size_t i = 0; i < M_ROWS; ++i) { // Iterates through rows of M (or elements of v)
        const double v_val = v[i]; // Cache v_val for inner loop for better performance
        for (size_t j = 0; j < M_COLS; ++j) { // Iterates through columns of M (or elements of result)
            result[j] += v_val * m(i, j); 
        }
    }
}

// Outer product for weight update: V_a * V_b (A_SIZE x 1 * 1 x B_SIZE -> A_SIZE x B_SIZE)
// Operates in-place on 'result'.
template <size_t A_SIZE, size_t B_SIZE>
void outer_product_in_place(const FixedVector<A_SIZE>& a, const FixedVector<B_SIZE>& b, FixedMatrix<A_SIZE, B_SIZE>& result) noexcept { 
    // Loop order chosen for cache efficiency in row-major matrix `result`.
    for (size_t i = 0; i < A_SIZE; ++i) { // Iterates through rows of result (or elements of a)
        const double a_val = a[i]; // Cache a_val for inner loop
        for (size_t j = 0; j < B_SIZE; ++j) { // Iterates through columns of result (or elements of b)
            result(i, j) = a_val * b[j]; 
        }
    }
}

// Element-wise subtraction (a - b) in-place into 'result'.
template <size_t N>
void subtract_elementwise_in_place(const FixedVector<N>& a, const FixedVector<N>& b, FixedVector<N>& result) noexcept {
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] - b[i];
    }
}

// Element-wise addition (a + b) in-place into 'result'.
template <size_t N>
void add_elementwise_in_place(const FixedVector<N>& a, const FixedVector<N>& b, FixedVector<N>& result) noexcept {
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] + b[i];
    }
}

// --- XORModel Class ---

// XORModel is templated on layer sizes, making all internal data structures fixed-size.
template <size_t InputSize, size_t HiddenSize, size_t OutputSize>
class XORModel {
private:
    FixedMatrix<InputSize, HiddenSize> hidden_weights; 
    FixedMatrix<HiddenSize, OutputSize> output_weights; 
    FixedVector<HiddenSize> hidden_biases;  
    FixedVector<OutputSize> output_biases;  
    
    // Intermediate storage for backprop. These are now fixed-size arrays,
    // eliminating all dynamic memory allocations during forward/backward passes.
    FixedVector<HiddenSize> hidden_input_net;   
    FixedVector<HiddenSize> hidden_output;      
    FixedVector<OutputSize> final_input_net;    
    FixedVector<OutputSize> output_delta_buffer; 
    // hidden_error_buffer holds the result of `output_delta * output_weights^T`
    FixedVector<HiddenSize> hidden_error_buffer; 
    FixedVector<HiddenSize> hidden_delta_buffer; 
    FixedMatrix<HiddenSize, OutputSize> output_weight_delta_buffer; 
    FixedMatrix<InputSize, HiddenSize> hidden_weight_delta_buffer; 

    std::mt19937 gen; // Seeded once with std::random_device for initial randomness
    std::normal_distribution<double> hidden_weight_distrib; // Glorot (Xavier) for Tanh
    std::normal_distribution<double> output_weight_distrib; // Glorot (Xavier) for Sigmoid

    /**
     * @brief Initializes weights with small random values using Glorot (Xavier) initialization.
     * @param m The matrix to initialize.
     * @param dist The normal distribution to draw values from.
     */
    template <size_t R, size_t C>
    void initialize_weights(FixedMatrix<R, C>& m, std::normal_distribution<double>& dist) {
        std::generate(m.data.begin(), m.data.end(), [&]() { return dist(gen); });
    }

    /**
     * @brief Initializes biases to zero.
     * @param v The vector to initialize.
     */
    template <size_t N>
    void initialize_biases(FixedVector<N>& v) noexcept {
        v.fill(0.0);
    }

    /**
     * @brief Performs element-wise operation (bias -= learning_rate * delta) on a bias vector.
     * @param bias The bias vector to update.
     * @param delta The delta vector for biases.
     * @param learning_rate The training rate.
     */
    template <size_t N>
    void update_biases_in_place(FixedVector<N>& bias, const FixedVector<N>& delta, double learning_rate) noexcept {
        for (size_t i = 0; i < N; ++i) {
            bias[i] -= learning_rate * delta[i];
        }
    }

    /**
     * @brief Updates a weight matrix using the weight delta and learning rate.
     * @param weights The weight matrix to update.
     * @param weight_delta The delta matrix for weights.
     * @param learning_rate The training rate.
     */
    template <size_t R, size_t C>
    void update_weights_in_place(FixedMatrix<R, C>& weights, const FixedMatrix<R, C>& weight_delta, double learning_rate) noexcept {
        for (size_t i = 0; i < R * C; ++i) {
            weights.data[i] -= learning_rate * weight_delta.data[i];
        }
    }

public:
    /**
     * @brief Constructor for the XORModel. Layer sizes are provided as template arguments.
     */
    XORModel() : 
        gen(std::random_device{}()), 
        // Glorot (Xavier) initialization: std::sqrt(1.0 / fan_in) is a common variant for tanh/sigmoid.
        // For hidden layer (tanh), fan_in is InputSize.
        hidden_weight_distrib(0.0, std::sqrt(1.0 / static_cast<double>(InputSize))), 
        // For output layer (sigmoid), fan_in is HiddenSize.
        output_weight_distrib(0.0, std::sqrt(1.0 / static_cast<double>(HiddenSize))) 
    {
        initialize_weights(hidden_weights, hidden_weight_distrib);
        initialize_weights(output_weights, output_weight_distrib);
        initialize_biases(hidden_biases);
        initialize_biases(output_biases);

        // Fixed-size buffers are default-initialized (all zeros) by their constructors.
        // No explicit 'resize' or 'assign' calls needed here, as their sizes are fixed at compile time.
    }

    /**
     * @brief Calculates the network's output for a given input (Forward Pass).
     * @param input_data The InputSize input vector.
     * @return The OutputSize output vector.
     */
    [[nodiscard]] FixedVector<OutputSize> forward(const FixedVector<InputSize>& input_data) {
        // Hidden Layer Input (net): Input * Hidden_Weights + Hidden_Biases
        // Dimensions: (1 x InputSize) * (InputSize x HiddenSize) -> (1 x HiddenSize)
        dot_in_place(input_data, hidden_weights, hidden_input_net);
        add_elementwise_in_place(hidden_input_net, hidden_biases, hidden_input_net);

        // Hidden Layer Output: Tanh(Hidden Layer Input)
        tanh_activation_in_place(hidden_input_net, hidden_output);
        
        // Final Layer Input (net): Hidden_Output * Output_Weights + Output_Biases
        // Dimensions: (1 x HiddenSize) * (HiddenSize x OutputSize) -> (1 x OutputSize)
        dot_in_place(hidden_output, output_weights, final_input_net);
        add_elementwise_in_place(final_input_net, output_biases, final_input_net);

        // Final Layer Output: Sigmoid(Final Layer Input)
        // Return by value, relying on RVO/NRVO or move semantics for efficiency,
        // which are highly optimized for std::array.
        FixedVector<OutputSize> final_output; 
        sigmoid_in_place(final_input_net, final_output);
        return final_output;
    }

    /**
     * @brief Adjusts weights and biases (Backpropagation).
     * @param input_data The InputSize input vector.
     * @param target The OutputSize target vector.
     * @param output The OutputSize predicted output vector from the forward pass.
     * @param learning_rate The training rate.
     */
    void backpropagate(const FixedVector<InputSize>& input_data, const FixedVector<OutputSize>& target, 
                       const FixedVector<OutputSize>& output, double learning_rate) noexcept {
        // 1. Calculate Output Error & Output Layer Delta (dJ/dout * dout/dnet)
        // Error = Output - Target (stored in output_delta_buffer)
        subtract_elementwise_in_place(output, target, output_delta_buffer); 
        
        // Output Delta = Error * Sigmoid_Derivative(Output)
        // Sigmoid derivative: output * (1 - output)
        for (size_t i = 0; i < OutputSize; ++i) {
            const double out_val = output[i];
            output_delta_buffer[i] *= (out_val * (1.0 - out_val));
        }

        // 2. Calculate Hidden Layer Error (Propagate output delta backward through weights)
        // This calculates `hidden_error_buffer = output_delta_buffer * output_weights^T`.
        // Given: output_delta_buffer is (1 x OutputSize) row vector.
        //        output_weights (W_HO) is FixedMatrix<HiddenSize, OutputSize> (H x O).
        //        W_HO^T is (O x H).
        // Operation: (1 x O) * (O x H) -> (1 x H).
        // Each element `hidden_error_buffer[h_idx]` is `sum_{o_idx} (output_delta_buffer[o_idx] * W_HO[h_idx][o_idx])`.
        hidden_error_buffer.fill(0.0); // Reset buffer to zero for accumulation
        
        for (size_t h_idx = 0; h_idx < HiddenSize; ++h_idx) { // Iterate over dimensions of hidden_error_buffer (H)
            for (size_t o_idx = 0; o_idx < OutputSize; ++o_idx) { // Iterate over dimensions of output_delta_buffer (O)
                // Access W_HO[h_idx][o_idx]
                hidden_error_buffer[h_idx] += output_delta_buffer[o_idx] * output_weights(h_idx, o_idx); 
            }
        }
        
        // Hidden Delta = Hidden_Error * Tanh_Derivative(Hidden_Output)
        // Tanh derivative: 1 - output^2
        for (size_t i = 0; i < HiddenSize; ++i) {
            const double h_out_val = hidden_output[i];
            hidden_delta_buffer[i] = hidden_error_buffer[i] * (1.0 - (h_out_val * h_out_val));
        }

        // 3. Update Output Layer Weights and Biases
        // Output Weight Delta = Transpose(Hidden_Output) * Output_Delta (H x 1 * 1 x O -> H x O)
        // This is equivalent to outer_product(hidden_output, output_delta_buffer)
        outer_product_in_place(hidden_output, output_delta_buffer, output_weight_delta_buffer);
        update_weights_in_place(output_weights, output_weight_delta_buffer, learning_rate);
        
        // Output Bias Delta = Output_Delta (element-wise subtraction, as delta IS the bias gradient)
        update_biases_in_place(output_biases, output_delta_buffer, learning_rate);

        // 4. Update Hidden Layer Weights and Biases
        // Hidden Weight Delta = Transpose(Input) * Hidden_Delta (I x 1 * 1 x H -> I x H)
        // This is equivalent to outer_product(input_data, hidden_delta_buffer)
        outer_product_in_place(input_data, hidden_delta_buffer, hidden_weight_delta_buffer);
        update_weights_in_place(hidden_weights, hidden_weight_delta_buffer, learning_rate);
        
        // Hidden Bias Delta = Hidden_Delta (element-wise subtraction, as delta IS the bias gradient)
        update_biases_in_place(hidden_biases, hidden_delta_buffer, learning_rate);
    }

    /**
     * @brief Trains the network for a specified number of epochs.
     * @param X The input data, a std::vector of FixedVectors.
     * @param y The target data, a std::vector of FixedVectors.
     * @param epochs The number of training epochs.
     * @param learning_rate The training rate.
     */
    void train(const std::vector<FixedVector<InputSize>>& X, const std::vector<FixedVector<OutputSize>>& y, int epochs, double learning_rate) {
        const size_t num_samples = X.size();
        std::cout << "Training for " << epochs << " epochs with LR=" << learning_rate << "..." << '\n';
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_squared_error = 0.0;
            
            for (size_t i = 0; i < num_samples; ++i) {
                // Forward pass - RVO is highly effective for std::array return.
                FixedVector<OutputSize> output = forward(X[i]); 
                
                // Calculate squared error
                for (size_t j = 0; j < OutputSize; ++j) {
                    double diff = output[j] - y[i][j];
                    total_squared_error += diff * diff;
                }
                
                // Backpropagate
                backpropagate(X[i], y[i], output, learning_rate);
            }
            
            // Report progress every 1000 epochs or on the first epoch.
            if (((epoch + 1) % 1000 == 0) || (epoch == 0)) { 
                std::cout << "Epoch " << (epoch + 1) << ", Total Squared Error: " 
                          << std::fixed << std::setprecision(8) << total_squared_error << '\n';
            }
        }
        std::cout << "Training complete." << '\n';
    }

    /**
     * @brief Computes the model's accuracy on the training data.
     * @param X The input data.
     * @param y The target data.
     * @return The percentage of correct predictions (0 or 1).
     */
    [[nodiscard]] double compute_accuracy(const std::vector<FixedVector<InputSize>>& X, const std::vector<FixedVector<OutputSize>>& y) {
        size_t correct_predictions = 0;
        const size_t num_samples = X.size();
        
        for (size_t i = 0; i < num_samples; ++i) {
            FixedVector<OutputSize> output = forward(X[i]);
            // Binary prediction: 0 if output < 0.5, 1 otherwise
            int predicted = (output[0] >= 0.5) ? 1 : 0;
            // Robustly convert target float to int for comparison (e.g., 0.0 -> 0, 1.0 -> 1)
            int actual = static_cast<int>(y[i][0] + 0.5); 
            
            if (predicted == actual) {
                correct_predictions++;
            }
        }
        
        return static_cast<double>(correct_predictions) / num_samples * 100.0;
    }
};

// --- Execution ---

int main() {
    // Optimize C++ standard streams for faster input/output operations.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // Data Definition for XOR.
    // Inputs are 2-element FixedVectors, Targets are 1-element FixedVectors.
    std::vector<FixedVector<2>> X = {{{0.0, 0.0}}, {{0.0, 1.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}};
    std::vector<FixedVector<1>> y = {{{0.0}}, {{1.0}}, {{1.0}}, {{0.0}}};

    // Model Instantiation: Input=2, Hidden=4, Output=1.
    // Sizes are now compile-time template parameters.
    XORModel<2, 4, 1> model;

    // Training: Increased epochs for robust convergence with Tanh activation and Glorot initialization.
    model.train(X, y, 20000, 0.1); 

    // Evaluation
    double accuracy = model.compute_accuracy(X, y);
    std::cout << "\n--- Evaluation ---" << '\n';
    std::cout << "Final Training Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << '\n';

    // Prediction
    std::cout << "\n--- Final Predictions ---" << '\n';
    const size_t num_test_samples = X.size(); // Cache size for main loop.
    for (size_t i = 0; i < num_test_samples; ++i) {
        FixedVector<1> output = model.forward(X[i]);
        int predicted = (output[0] >= 0.5) ? 1 : 0;
        std::cout << "Input (" << X[i][0] << ", " << X[i][1] << "): Predicted=" << predicted 
                  << " (Raw: " << std::fixed << std::setprecision(4) << output[0] << "), Target=" << static_cast<int>(y[i][0] + 0.5) << '\n';
    }

    return 0;
}
