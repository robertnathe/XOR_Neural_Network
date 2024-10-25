#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Define the XOR input and output data
std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

// Define the neural network model
class XORModel {
public:
    XORModel(size_t input_size, size_t hidden_size, size_t output_size) 
        : hidden_weights(hidden_size, std::vector<double>(input_size)),
          hidden_biases(hidden_size, 0.0),
          output_weights(output_size, std::vector<double>(hidden_size)),
          output_biases(output_size, 0.0),
          hidden_outputs(hidden_size) { // Initialize hidden_outputs here
        // Initialize weights randomly with a fixed seed
        std::mt19937 gen(1234); // Change the seed here for different initialization
        std::uniform_real_distribution<> distr(-0.5, 0.5); // Adjusted range for weight initialization
        for (auto& weights : hidden_weights) {
            for (auto& weight : weights) {
                weight = distr(gen);
            }
        }
        for (auto& weights : output_weights) {
            for (auto& weight : weights) {
                weight = distr(gen);
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        for (size_t i = 0; i < hidden_weights.size(); ++i) {
            double sum = hidden_biases[i];
            for (size_t j = 0; j < input.size(); ++j) {
                sum += hidden_weights[i][j] * input[j];
            }
            hidden_outputs[i] = hidden_activation(sum);
        }
        std::vector<double> output_outputs(output_weights.size());
        for (size_t i = 0; i < output_weights.size(); ++i) {
            double sum = output_biases[i];
            for (size_t j = 0; j < hidden_outputs.size(); ++j) {
                sum += output_weights[i][j] * hidden_outputs[j];
            }
            output_outputs[i] = activation(sum);
        }
        return output_outputs;
    }

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               size_t epochs, double learning_rate) {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const std::vector<double>& input = inputs[i];
                const std::vector<double>& target = targets[i];
                std::vector<double> outputs = forward(input);
                total_error += calculate_error(outputs, target);
                backpropagate(input, target, outputs, learning_rate);
            }
            std::cout << "Epoch " << epoch + 1 << ": Total error = " << total_error << std::endl;
        }
    }

    double compute_accuracy() {
        int correct_predictions = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<double> output = forward(X[i]);
            int predicted = output[0] >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(y[i][0]);
            correct_predictions += (predicted == actual);
        }
        return static_cast<double>(correct_predictions) / static_cast<double>(X.size()) * 100.0; // Return accuracy as a percentage
    }

private:
    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x)); // Sigmoid for output layer
    }

    double hidden_activation(double x) {
        return x < 0 ? 0 : x; // ReLU for hidden layer
    }

    double calculate_error(const std::vector<double>& outputs, const std::vector<double>& targets) {
        double error = 0.0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            error += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
        }
        return error / 2.0;
    }

    void backpropagate(const std::vector<double>& input, const std::vector<double>& target,
                       const std::vector<double>& outputs, double learning_rate) {
        // Calculate output layer error and gradients
        std::vector<double> output_errors(output_weights.size());
        for (size_t i = 0; i < output_weights.size(); ++i) {
            output_errors[i] = (outputs[i] - target[i]) * outputs[i] * (1.0 - outputs[i]);
        }
        // Update output layer weights and biases
        for (size_t i = 0; i < output_weights.size(); ++i) {
            for (size_t j = 0; j < hidden_outputs.size(); ++j) {
                output_weights[i][j] -= learning_rate * output_errors[i] * hidden_outputs[j];
            }
            output_biases[i] -= learning_rate * output_errors[i];
        }
        // Calculate hidden layer error and gradients
        std::vector<double> hidden_errors(hidden_weights.size());
        for (size_t i = 0; i < hidden_weights.size(); ++i) {
            hidden_errors[i] = 0.0;
            for (size_t j = 0; j < output_weights.size(); ++j) {
                hidden_errors[i] += output_errors[j] * output_weights[j][i];
            }
            hidden_errors[i] *= hidden_outputs[i] * (1.0 - hidden_outputs[i]); // Sigmoid derivative
        }
        // Update hidden layer weights and biases
        for (size_t i = 0; i < hidden_weights.size(); ++i) {
            for (size_t j = 0; j < input.size(); ++j) {
                hidden_weights[i][j] -= learning_rate * hidden_errors[i] * input[j];
            }
            hidden_biases[i] -= learning_rate * hidden_errors[i];
        }
    }

    std::vector<std::vector<double>> hidden_weights;
    std::vector<double> hidden_biases;
    std::vector<std::vector<double>> output_weights;
    std::vector<double> output_biases;
    std::vector<double> hidden_outputs; // Add hidden_outputs as a member variable
};

int main() {
    XORModel model(2, 3, 1); // Increased hidden layer size
    model.train(X, y, 10000, 0.05); // Changed epochs and learning rate
    // Evaluate the model
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> output = model.forward(X[i]);
        std::cout << "Input: " << X[i][0] << " " << X[i][1] << " => Predicted Output: " << output[0] << ", Actual Output: " << y[i][0] << std::endl;
    }
    // Calculate and print accuracy
    double accuracy = model.compute_accuracy();
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}

/*
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
// Define the XOR input and output data
std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

// Define the neural network model
class XORModel {
public:
    XORModel(size_t input_size, size_t hidden_size, size_t output_size) 
        : hidden_weights(hidden_size, std::vector<double>(input_size)),
          hidden_biases(hidden_size, 0.0),
          output_weights(output_size, std::vector<double>(hidden_size)),
          output_biases(output_size, 0.0) {
        // Initialize weights randomly with a fixed seed
        std::mt19937 gen(1234); // Change the seed here for different initialization
        std::uniform_real_distribution<> distr(-0.5, 0.5); // Adjusted range for weight initialization
        for (auto& weights : hidden_weights) {
            for (auto& weight : weights) {
                weight = distr(gen);
            }
        }
        for (auto& weights : output_weights) {
            for (auto& weight : weights) {
                weight = distr(gen);
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        hidden_outputs.resize(hidden_weights.size());
        for (size_t i = 0; i < hidden_weights.size(); ++i) {
            double sum = hidden_biases[i];
            for (size_t j = 0; j < input.size(); ++j) {
                sum += hidden_weights[i][j] * input[j];
            }
            hidden_outputs[i] = hidden_activation(sum);
        }
        std::vector<double> output_outputs(output_weights.size());
        for (size_t i = 0; i < output_weights.size(); ++i) {
            double sum = output_biases[i];
            for (size_t j = 0; j < hidden_outputs.size(); ++j) {
                sum += output_weights[i][j] * hidden_outputs[j];
            }
            output_outputs[i] = activation(sum);
        }
        return output_outputs;
    }

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               size_t epochs, double learning_rate) {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const std::vector<double>& input = inputs[i];
                const std::vector<double>& target = targets[i];
                std::vector<double> outputs = forward(input);
                total_error += calculate_error(outputs, target);
                backpropagate(input, target, outputs, learning_rate);
            }
            std::cout << "Epoch " << epoch + 1 << ": Total error = " << total_error << std::endl;
        }
    }

    double compute_accuracy() {
        int correct_predictions = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<double> output = forward(X[i]);
            int predicted = output[0] >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(y[i][0]);
            correct_predictions += (predicted == actual);
        }
        return static_cast<double>(correct_predictions) / static_cast<double>(X.size()) * 100.0; // Return accuracy as a percentage
    }

private:
    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x)); // Sigmoid for output layer
    }

    double hidden_activation(double x) {
        return x < 0 ? 0 : x; // ReLU for hidden layer
    }

    double calculate_error(const std::vector<double>& outputs, const std::vector<double>& targets) {
        double error = 0.0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            error += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
        }
        return error / 2.0;
    }

    void backpropagate(const std::vector<double>& input, const std::vector<double>& target,
                       const std::vector<double>& outputs, double learning_rate) {
        // Calculate output layer error and gradients
        std::vector<double> output_errors(output_weights.size());
        for (size_t i = 0; i < output_weights.size(); ++i) {
            output_errors[i] = (outputs[i] - target[i]) * outputs[i] * (1.0 - outputs[i]);
        }
        // Update output layer weights and biases
        for (size_t i = 0; i < output_weights.size(); ++i) {
            for (size_t j = 0; j < hidden_outputs.size(); ++j) {
                output_weights[i][j] -= learning_rate * output_errors[i] * hidden_outputs[j];
            }
            output_biases[i] -= learning_rate * output_errors[i];
        }
        // Calculate hidden layer error and gradients
        std::vector<double> hidden_errors(hidden_weights.size());
        for (size_t i = 0; i < hidden_weights.size(); ++i) {
            hidden_errors[i] = 0.0;
            for (size_t j = 0; j < output_weights.size(); ++j) {
                hidden_errors[i] += output_errors[j] * output_weights[j][i];
            }
            hidden_errors[i] *= hidden_outputs[i] * (1.0 - hidden_outputs[i]); // Sigmoid derivative
        }
        // Update hidden layer weights and biases
        for (size_t i = 0; i < hidden_weights.size(); ++i) {
            for (size_t j = 0; j < input.size(); ++j) {
                hidden_weights[i][j] -= learning_rate * hidden_errors[i] * input[j];
            }
            hidden_biases[i] -= learning_rate * hidden_errors[i];
        }
    }

    std::vector<std::vector<double>> hidden_weights;
    std::vector<double> hidden_biases;
    std::vector<std::vector<double>> output_weights;
    std::vector<double> output_biases;
    std::vector<double> hidden_outputs; // Add hidden_outputs as a member variable
};

int main() {
    XORModel model(2, 3, 1); // Increased hidden layer size
    model.train(X, y, 10000, 0.05); // Changed epochs and learning rate
    // Evaluate the model
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> output = model.forward(X[i]);
        std::cout << "Input: " << X[i][0] << " " << X[i][1] << " => Predicted Output: " << output[0] << ", Actual Output: " << y[i][0] << std::endl;
    }
    // Calculate and print accuracy
    double accuracy = model.compute_accuracy();
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
*/
