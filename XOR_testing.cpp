#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>

const size_t INPUT_SIZE = 2;
const size_t HIDDEN_SIZE = 3;
const size_t OUTPUT_SIZE = 1;
const size_t EPOCHS = 10000;
const double LEARNING_RATE = 0.05;

// Define the XOR input and output data
double X[4][INPUT_SIZE] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double y[4][OUTPUT_SIZE] = {{0}, {1}, {1}, {0}};

// Define the neural network model
class XORModel {
public:
    XORModel() {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<> distr(-0.5, 0.5);

        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            for (size_t j = 0; j < INPUT_SIZE; ++j) {
                hidden_weights[i][j] = distr(gen);
            }
            hidden_biases[i] = 0.0;
        }

        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                output_weights[i][j] = distr(gen);
            }
            output_biases[i] = 0.0;
        }
    }

    void forward(const double input[INPUT_SIZE], double output[OUTPUT_SIZE]) {
        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            double sum = hidden_biases[i];
            for (size_t j = 0; j < INPUT_SIZE; ++j) {
                sum += hidden_weights[i][j] * input[j];
            }
            hidden_outputs[i] = hidden_activation(sum);
        }

        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            double sum = output_biases[i];
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                sum += output_weights[i][j] * hidden_outputs[j];
            }
            output[i] = activation(sum);
        }
    }

    void train() {
        std::vector<double> errors; // Store total errors for each epoch
        for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < 4; ++i) {
                const double* input = X[i];
                const double* target = y[i];
                double outputs[OUTPUT_SIZE];
                forward(input, outputs);
                total_error += calculate_error(outputs, target);
                backpropagate(input, target, outputs);
            }
            errors.push_back(total_error); // Store the error for this epoch
            std::cout << "Epoch " << epoch + 1 << ": Total error = " << total_error << std::endl;
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
            double output[OUTPUT_SIZE];
            forward(X[i], output);
            int predicted = output[0] >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(y[i][0]);
            correct_predictions += (predicted == actual);
        }
        return static_cast<double>(correct_predictions) / 4.0 * 100.0;
    }

private:
    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x)); // Sigmoid for output layer
    }

    double hidden_activation(double x) {
        return x < 0 ? 0 : x; // ReLU for hidden layer
    }

    double calculate_error(const double outputs[OUTPUT_SIZE], const double targets[OUTPUT_SIZE]) {
        double error = 0.0;
        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            error += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
        }
        return error / 2.0;
    }

    void backpropagate(const double input[INPUT_SIZE], const double target[OUTPUT_SIZE], const double outputs[OUTPUT_SIZE]) {
        double output_errors[OUTPUT_SIZE];
        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            output_errors[i] = (outputs[i] - target[i]) * outputs[i] * (1.0 - outputs[i]);
        }

        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                output_weights[i][j] -= LEARNING_RATE * output_errors[i] * hidden_outputs[j];
            }
            output_biases[i] -= LEARNING_RATE * output_errors[i];
        }

        double hidden_errors[HIDDEN_SIZE];
        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            hidden_errors[i] = 0.0;
            for (size_t j = 0; j < OUTPUT_SIZE; ++j) {
                hidden_errors[i] += output_errors[j] * output_weights[j][i];
            }
            hidden_errors[i] *= hidden_outputs[i] * (1.0 - hidden_outputs[i]); // ReLU derivative
        }

        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            for (size_t j = 0; j < INPUT_SIZE; ++j) {
                hidden_weights[i][j] -= LEARNING_RATE * hidden_errors[i] * input[j];
            }
            hidden_biases[i] -= LEARNING_RATE * hidden_errors[i];
        }
    }

    double hidden_weights[HIDDEN_SIZE][INPUT_SIZE];
    double hidden_biases[HIDDEN_SIZE];
    double output_weights[OUTPUT_SIZE][HIDDEN_SIZE];
    double output_biases[OUTPUT_SIZE];
    double hidden_outputs[HIDDEN_SIZE];
};

void test_on_new_data(XORModel& model) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> distr(0.0, 1.0);

    for (size_t i = 0; i < 5; ++i) { // Generate 5 random inputs
        double input[INPUT_SIZE] = {distr(gen), distr(gen)};
        double output[OUTPUT_SIZE];
        model.forward(input, output);
        std::cout << "Random Input: " << input[0] << ", " << input[1]
                  << " => Predicted Output: " << output[0] << std::endl;
    }
}

int main() {
    XORModel model;
    model.train();

    // Evaluate the model
    for (size_t i = 0; i < 4; ++i) {
        double output[OUTPUT_SIZE];
        model.forward(X[i], output);
        std::cout << "Input: " << X[i][0] << " " << X[i][1] << " => Predicted Output: " << output[0] << ", Actual Output: " << y[i][0] << std::endl;
    }

    // Calculate and print accuracy
    double accuracy = model.compute_accuracy();
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    // Test on new random data
    test_on_new_data(model);

    return 0;
}

/*
#include <iostream>
#include <cmath>
#include <random>

const size_t INPUT_SIZE = 2;
const size_t HIDDEN_SIZE = 3;
const size_t OUTPUT_SIZE = 1;
const size_t EPOCHS = 2000;
const double LEARNING_RATE = 0.05;

// Define the XOR input and output data
double X[4][INPUT_SIZE] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double y[4][OUTPUT_SIZE] = {{0}, {1}, {1}, {0}};

// Define the neural network model
class XORModel {
public:
    XORModel() {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<> distr(-0.5, 0.5);

        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            for (size_t j = 0; j < INPUT_SIZE; ++j) {
                hidden_weights[i][j] = distr(gen);
            }
            hidden_biases[i] = 0.0;
        }

        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                output_weights[i][j] = distr(gen);
            }
            output_biases[i] = 0.0;
        }
    }

    void forward(const double input[INPUT_SIZE], double output[OUTPUT_SIZE]) {
        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            double sum = hidden_biases[i];
            for (size_t j = 0; j < INPUT_SIZE; ++j) {
                sum += hidden_weights[i][j] * input[j];
            }
            hidden_outputs[i] = hidden_activation(sum);
        }

        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            double sum = output_biases[i];
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                sum += output_weights[i][j] * hidden_outputs[j];
            }
            output[i] = activation(sum);
        }
    }

    void train() {
        for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < 4; ++i) {
                const double* input = X[i];
                const double* target = y[i];
                double outputs[OUTPUT_SIZE];
                forward(input, outputs);
                total_error += calculate_error(outputs, target);
                backpropagate(input, target, outputs);
            }
            std::cout << "Epoch " << epoch + 1 << ": Total error = " << total_error << std::endl;
        }
    }

    double compute_accuracy() {
        int correct_predictions = 0;
        for (size_t i = 0; i < 4; ++i) {
            double output[OUTPUT_SIZE];
            forward(X[i], output);
            int predicted = output[0] >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(y[i][0]);
            correct_predictions += (predicted == actual);
        }
        return static_cast<double>(correct_predictions) / 4.0 * 100.0;
    }

private:
    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x)); // Sigmoid for output layer
    }

    double hidden_activation(double x) {
        return x < 0 ? 0 : x; // ReLU for hidden layer
    }

    double calculate_error(const double outputs[OUTPUT_SIZE], const double targets[OUTPUT_SIZE]) {
        double error = 0.0;
        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            error += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
        }
        return error / 2.0;
    }

    void backpropagate(const double input[INPUT_SIZE], const double target[OUTPUT_SIZE], const double outputs[OUTPUT_SIZE]) {
        double output_errors[OUTPUT_SIZE];
        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            output_errors[i] = (outputs[i] - target[i]) * outputs[i] * (1.0 - outputs[i]);
        }

        for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
            for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
                output_weights[i][j] -= LEARNING_RATE * output_errors[i] * hidden_outputs[j];
            }
            output_biases[i] -= LEARNING_RATE * output_errors[i];
        }

        double hidden_errors[HIDDEN_SIZE];
        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            hidden_errors[i] = 0.0;
            for (size_t j = 0; j < OUTPUT_SIZE; ++j) {
                hidden_errors[i] += output_errors[j] * output_weights[j][i];
            }
            hidden_errors[i] *= hidden_outputs[i] * (1.0 - hidden_outputs[i]); // ReLU derivative
        }

        for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
            for (size_t j = 0; j < INPUT_SIZE; ++j) {
                hidden_weights[i][j] -= LEARNING_RATE * hidden_errors[i] * input[j];
            }
            hidden_biases[i] -= LEARNING_RATE * hidden_errors[i];
        }
    }

    double hidden_weights[HIDDEN_SIZE][INPUT_SIZE];
    double hidden_biases[HIDDEN_SIZE];
    double output_weights[OUTPUT_SIZE][HIDDEN_SIZE];
    double output_biases[OUTPUT_SIZE];
    double hidden_outputs[HIDDEN_SIZE];
};

void test_on_new_data(XORModel& model) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> distr(0.0, 1.0);

    for (size_t i = 0; i < 5; ++i) { // Generate 5 random inputs
        double input[INPUT_SIZE] = {distr(gen), distr(gen)};
        double output[OUTPUT_SIZE];
        model.forward(input, output);
        std::cout << "Random Input: " << input[0] << ", " << input[1]
                  << " => Predicted Output: " << output[0] << std::endl;
    }
}

int main() {
    XORModel model;
    model.train();

    // Evaluate the model
    for (size_t i = 0; i < 4; ++i) {
        double output[OUTPUT_SIZE];
        model.forward(X[i], output);
        std::cout << "Input: " << X[i][0] << " " << X[i][1] << " => Predicted Output: " << output[0] << ", Actual Output: " << y[i][0] << std::endl;
    }

    // Calculate and print accuracy
    double accuracy = model.compute_accuracy();
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    // Test on new random data
    test_on_new_data(model);

    return 0;
}
*/
