#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm> // For std::transform

// Use arrays instead of vectors for fixed-size data
double X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double y[4] = {0, 1, 1, 0};

// Define the neural network model
class XORModel {
public:
    XORModel(size_t input_size, size_t hidden_size, size_t output_size)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size) {

        // Allocate memory dynamically for weights and biases
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

        // Initialize weights and biases
        std::mt19937 gen(1234);
        std::uniform_real_distribution<> distr(-0.5, 0.5);

        for (size_t i = 0; i < hidden_size_; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                hidden_weights_[i][j] = distr(gen);
            }
            hidden_biases_[i] = 0.0;
        }
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < hidden_size_; ++j) {
                output_weights_[i][j] = distr(gen);
            }
            output_biases_[i] = 0.0;
        }
    }

    ~XORModel() {
        // Deallocate dynamically allocated memory
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
        // Hidden layer
        for (size_t i = 0; i < hidden_size_; ++i) {
            double sum = hidden_biases_[i];
            for (size_t j = 0; j < input_size_; ++j) {
                sum += hidden_weights_[i][j] * input[j];
            }
            hidden_outputs_[i] = hidden_activation(sum);
        }

        // Output layer
        double output_output = 0.0;
        for (size_t i = 0; i < output_size_; ++i) {
             output_output = output_biases_[i];
            for (size_t j = 0; j < hidden_size_; ++j) {
                output_output += output_weights_[i][j] * hidden_outputs_[j];
            }
             output_output = activation(output_output);
        }
        return output_output;
    }

    void train(double inputs[][2], double* targets, size_t epochs, double learning_rate) {
		std::vector<double> errors; // Store total errors for each epoch
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < 4; ++i) {
                double output = forward(inputs[i]);
                total_error += calculate_error(output, targets[i]);
                backpropagate(inputs[i], targets[i], output, learning_rate);
            }
            if (epoch % 100 == 0)
            {
                std::cout << "Epoch " << epoch + 1 << ": Total error = " << total_error << std::endl;
                errors.push_back(total_error); // Store the error for this epoch
            }
        }
        // Write errors to a file
        std::ofstream error_file("errors.txt");
        if (error_file.is_open()) {
            for (const auto& error : errors) {
                error_file << error << std::endl;
            }
            error_file.close();
        } else {
            std::cerr << "Unable to open file for writing errors." << std::endl;
        }
    }

    double compute_accuracy() {
        int correct_predictions = 0;
        for (size_t i = 0; i < 4; ++i) {
            double output = forward(X[i]);
            int predicted = output >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(y[i]);
            correct_predictions += (predicted == actual);
        }
        return static_cast<double>(correct_predictions) / 4.0 * 100.0;
    }

private:
    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double hidden_activation(double x) {
        return x < 0 ? 0 : x;
    }
    
    double activation_derivative(double x) {
        return x * (1.0 - x);
    }

    double calculate_error(double output, double target) {
        return 0.5 * (output - target) * (output - target);
    }

    void backpropagate(const double* input, double target, double output, double learning_rate) {
        // Output layer error and gradient
        double output_error = (output - target) * activation_derivative(output);

        // Update output layer weights and biases
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < hidden_size_; ++j) {
                output_weights_[i][j] -= learning_rate * output_error * hidden_outputs_[j];
            }
            output_biases_[i] -= learning_rate * output_error;
        }

        // Hidden layer error and gradient
        for (size_t i = 0; i < hidden_size_; ++i) {
            double hidden_error = 0.0;
            for (size_t j = 0; j < output_size_; ++j) {
                hidden_error += output_error * output_weights_[j][i];
            }
            hidden_error *= activation_derivative(hidden_outputs_[i]); // Derivative of ReLU: 1 if x > 0, 0 otherwise

            // Update hidden layer weights and biases
            for (size_t j = 0; j < input_size_; ++j) {
                hidden_weights_[i][j] -= learning_rate * hidden_error * input[j];
            }
            hidden_biases_[i] -= learning_rate * hidden_error;
        }
    }

    size_t input_size_;
    size_t hidden_size_;
    size_t output_size_;
    double** hidden_weights_;
    double* hidden_biases_;
    double** output_weights_;
    double* output_biases_;
    double* hidden_outputs_;
};

int main() {
    // Experiment with different learning rates and hidden layer sizes
    double learning_rates[] = {0.02};
    int hidden_layer_sizes[] = {20, 50, 100};
    for (double learning_rate : learning_rates) {
        for (int hidden_layer_size : hidden_layer_sizes) {
            XORModel model(2, hidden_layer_size, 1);
            model.train(X, y, 10000, learning_rate);
            // Evaluate the model
            for (size_t i = 0; i < 4; ++i) {
                double output = model.forward(X[i]);
                std::cout << "Input: " << X[i][0] << " " << X[i][1] << " => Predicted Output: " << output << ", Actual Output: " << y[i] << std::endl;
            }
            // Calculate and print accuracy
            double accuracy = model.compute_accuracy();
            std::cout << "Accuracy: " << accuracy << "%" << std::endl;
            std::cout << "Learning rate: " << learning_rate << ", Hidden layer size: " << hidden_layer_size << std::endl;
        }
    }
    return 0;
}
