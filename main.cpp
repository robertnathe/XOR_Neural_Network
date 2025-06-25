#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

double X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double y[4] = {0, 1, 1, 0};

class XORModel {
public:
    XORModel(size_t input_size, size_t hidden_size, size_t output_size)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size) {
        
        // Allocate memory for weights and biases
        hidden_weights_ = new double*[hidden_size_];
        for (size_t i = 0; i < hidden_size_; ++i) {
            hidden_weights_[i] = new double[input_size_];
        }
        hidden_biases_ = new double[hidden_size_];
        output_weights_ = new double*[output_size_];
        for (size_t i = 0; i < output_size_; ++i) {
            output_weights_[i] = new double[hidden_size_];
        }
        output_biases_ = new double[output_size_];
        hidden_outputs_ = new double[hidden_size_];

        // Initialize weights and biases with He/Xavier initialization
        std::mt19937 gen(1234);
        
        // He initialization for ReLU hidden layer
        double stddev_h = std::sqrt(2.0 / input_size_);
        std::normal_distribution<> distr_h(0.0, stddev_h);
        for (size_t i = 0; i < hidden_size_; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                hidden_weights_[i][j] = distr_h(gen);
            }
            hidden_biases_[i] = 0.0;
        }
        
        // Xavier initialization for sigmoid output layer
        double stddev_o = std::sqrt(1.0 / hidden_size_);
        std::normal_distribution<> distr_o(0.0, stddev_o);
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < hidden_size_; ++j) {
                output_weights_[i][j] = distr_o(gen);
            }
            output_biases_[i] = 0.0;
        }
    }

    ~XORModel() {
        // Deallocate memory
        for (size_t i = 0; i < hidden_size_; ++i) {
            delete[] hidden_weights_[i];
        }
        delete[] hidden_weights_;
        delete[] hidden_biases_;
        
        for (size_t i = 0; i < output_size_; ++i) {
            delete[] output_weights_[i];
        }
        delete[] output_weights_;
        delete[] output_biases_;
        delete[] hidden_outputs_;
    }

    double forward(const double* input) {
        // Hidden layer computation
        for (size_t i = 0; i < hidden_size_; ++i) {
            double sum = hidden_biases_[i];
            for (size_t j = 0; j < input_size_; ++j) {
                sum += hidden_weights_[i][j] * input[j];
            }
            hidden_outputs_[i] = hidden_activation(sum);
        }

        // Output layer computation
        double output_sum = output_biases_[0];
        for (size_t j = 0; j < hidden_size_; ++j) {
            output_sum += output_weights_[0][j] * hidden_outputs_[j];
        }
        return activation(output_sum);
    }

    void train(double inputs[][2], double* targets, size_t epochs, double learning_rate) {
        std::vector<double> errors;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < 4; ++i) {
                double output = forward(inputs[i]);
                total_error += calculate_error(output, targets[i]);
                backpropagate(inputs[i], targets[i], output, learning_rate);
            }
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ": Error = " << total_error << std::endl;
                errors.push_back(total_error);
            }
        }
    }

    double compute_accuracy() {
        int correct = 0;
        for (int i = 0; i < 4; ++i) {
            double output = forward(X[i]);
            if ((output >= 0.5) == (y[i] >= 0.5)) correct++;
        }
        return (correct / 4.0) * 100.0;
    }

private:
    size_t input_size_;
    size_t hidden_size_;
    size_t output_size_;
    double** hidden_weights_;
    double* hidden_biases_;
    double** output_weights_;
    double* output_biases_;
    double* hidden_outputs_;

    // Activation functions
    double activation(double x) {  // Sigmoid
        return 1.0 / (1.0 + std::exp(-x));
    }

    double hidden_activation(double x) {  // ReLU
        return x > 0 ? x : 0;
    }
    
    double activation_derivative(double x) {  // Sigmoid derivative
        return x * (1.0 - x);
    }

    double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    double calculate_error(double output, double target) {
        return 0.5 * std::pow(output - target, 2);
    }

    void backpropagate(const double* input, double target, double output, double learning_rate) {
        // Output layer error
        double output_error = (output - target) * activation_derivative(output);

        // Update output weights and biases
        for (size_t j = 0; j < hidden_size_; ++j) {
            output_weights_[0][j] -= learning_rate * output_error * hidden_outputs_[j];
        }
        output_biases_[0] -= learning_rate * output_error;

        // Hidden layer error (using ReLU derivative)
        for (size_t i = 0; i < hidden_size_; ++i) {
            double hidden_error = output_error * output_weights_[0][i];
            hidden_error *= relu_derivative(hidden_outputs_[i]);
            
            // Update hidden weights and biases
            for (size_t j = 0; j < input_size_; ++j) {
                hidden_weights_[i][j] -= learning_rate * hidden_error * input[j];
            }
            hidden_biases_[i] -= learning_rate * hidden_error;
        }
    }
};

int main() {
    // Experiment with parameters
    double learning_rates[] = {0.02};
    size_t hidden_sizes[] = {20, 50, 100};
    
    for (double lr : learning_rates) {
        for (size_t hs : hidden_sizes) {
            XORModel model(2, hs, 1);
            model.train(X, y, 3000, lr);
            
            // Test predictions
            for (int i = 0; i < 4; ++i) {
                double pred = model.forward(X[i]);
                std::cout << X[i][0] << " XOR " << X[i][1] << " = " 
                          << pred << " (expected " << y[i] << ")\n";
            }
            
            std::cout << "Accuracy: " << model.compute_accuracy() << "%\n";
            std::cout << "Params: lr=" << lr << ", hidden_size=" << hs << "\n\n";
        }
    }
    return 0;
}
