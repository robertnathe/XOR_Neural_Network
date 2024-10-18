// Author: Robert Nathe
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
using namespace std;
// Hyperparameters for the neural network
float learning_rate = 1.414213562; // Learning rate (sqrt of 2, can be adjusted)
float momentum = 0.25; // Momentum factor for weight updates
// Weights for the neural network (9 weights for 2 inputs and 2 hidden neurons)
float weights[9] = {}; 
// Training data (inputs) for the neural network
double training_data[4][2] = { 
    { 1, 0 },
    { 1, 1 },
    { 0, 1 },
    { 0, 0 } 
};
// Expected output (targets) for the training data
int answer_data[4] = { 1, 0, 1, 0 }; // Correct outputs for each input combination
// Additional variables for calculations
int bias = 1; // Bias term for the neurons
float h1; // Output of the first hidden neuron
float h2; // Output of the second hidden neuron
float error[4]; // Error for each training example
float output_neuron; // Final output of the network
float gradients[9]; // Gradients for weight updates
float derivative_O1; // Derivative for output neuron
float derivative_h1; // Derivative for first hidden neuron
float derivative_h2; // Derivative for second hidden neuron
float sum_output; // Weighted sum for output neuron
float sum_h1; // Weighted sum for first hidden neuron
float sum_h2; // Weighted sum for second hidden neuron
float update_weights[9]; // Temporary storage for weight updates
float prev_weight_update[9] = { 0 }; // Previous weight updates for momentum
float RMSE_ERROR = 1; // Root Mean Square Error
int epoch = 0; // Current epoch count
float RMSE_array_error[20000]; // Array to store RMSE for each epoch
float user_input[2]; // User input for testing the network
char choice = 'Y'; // User choice for input repetition
bool repeat = false; // Flag to indicate if training should repeat
//////// Function Prototypes ////////
void train_for_epoch();
bool should_restart_training();
void restart_training();
void reset_weight_updates();
void reset_errors();
void reset_rmse_history();
void train_neural_net();
float sigmoid_function(float x);
void calc_hidden_layers(int x);
void calc_output_neuron();
void calc_error(int x);
void calc_derivatives(int x);
void calc_gradient(int x);
void calc_updates();
void update_new_weights();
void calc_RMSE_ERROR();
void generate_weights();
void train_neural_net();
void get_user_input();
void calculate_outputs();
void start_input();
void save_to_file(const string& filename, const float* data, int count);
void save_weights_to_file(const string& filename, const float* weights, int count);
void safe_data();
//////// Function Prototypes ////////
int main()
{
    generate_weights(); // Initialize weights randomly
    train_neural_net(); // Train the neural network
    safe_data(); // Save error and weight data to files
    start_input(); // Allow user to input values for prediction
    system("pause"); // Pause the console (Windows specific)
}
// Helper function to save RMSE data to a file
void save_to_file(const string& filename, const float* data, int count)
{
    ofstream file(filename);
    if (!file) {
        cerr << "Error opening file: " << filename << endl; // Error handling
        return;
    }
    
    for (int i = 0; i < count; i++)
    {
        file << i << "   " << data[i] << endl; // Log index and corresponding value
    }
    file.close(); // Close the file
}
// Helper function to save weights to a file
void save_weights_to_file(const string& filename, const float* weights, int count)
{
    ofstream file(filename);
    if (!file) {
        cerr << "Error opening file: " << filename << endl; // Error handling
        return;
    }
    for (int i = 0; i < count; i++)
    {
        file << i << "   " << weights[i] << endl; // Log weight index and value
    }
    file.close(); // Close the file
}
void safe_data()
{
    // Save RMSE data to a file
    save_to_file("errorData1.txt", RMSE_array_error, epoch);
    // Save weights to a file
    save_weights_to_file("weight_data1.txt", weights, 9);
}
// Function to get user input
void get_user_input()
{
    cout << "Enter data 1: "; cin >> user_input[0]; 
    cout << "Enter data 2: "; cin >> user_input[1]; 
}
// Function to calculate outputs of the neurons
void calculate_outputs()
{
    // Calculate sums for hidden neurons
    sum_h1 = (user_input[0] * weights[0]) + (user_input[1] * weights[2]) + (bias * weights[4]);
    sum_h2 = (user_input[0] * weights[1]) + (user_input[1] * weights[3]) + (bias * weights[5]);
    // Apply sigmoid activation function
    h1 = sigmoid_function(sum_h1);
    h2 = sigmoid_function(sum_h2);
    // Calculate output neuron value
    sum_output = (h1 * weights[6]) + (h2 * weights[7]) + (bias * weights[8]);
    output_neuron = sigmoid_function(sum_output);
}
void start_input()
{
    // Loop to allow user to input data for predictions
    while (true)
    {
        // Check if the user wants to continue
        if (choice != 'Y' && choice != 'y') {
            cout << "Thank you for using the prediction system. Goodbye!" << endl; // Informative exit message
            break; // Exit if user chooses not to continue
        }        
        // Get user input
        get_user_input();
        // Calculate hidden neuron outputs and the final output
        calculate_outputs();
        // Display the output
        cout << "Output = " << output_neuron << endl;
        // Ask user if they want to input again
        cout << "Again? (Y/N): "; 
        cin >> choice;
    }
}
// Function to train the network for one epoch
void train_for_epoch()
{
    for (int i = 0; i < 4; i++)
    {
        calc_hidden_layers(i); // Calculate hidden layer outputs
        calc_output_neuron(); // Calculate output neuron value
        calc_error(i); // Calculate error for this training example
        calc_derivatives(i); // Calculate derivatives for backpropagation
        calc_gradient(i); // Calculate gradients for weight updates
        calc_updates(); // Prepare weight updates
        update_new_weights(); // Update weights
    }
}
// Function to check if training should be restarted
bool should_restart_training()
{
    return (epoch > 4000 && RMSE_ERROR > 0.5);
}
// Function to reset training parameters and reinitialize weights
void restart_training()
{
    repeat = true; // Set flag to indicate restart
    reset_weight_updates();
    reset_errors();
    reset_rmse_history();
    epoch = 0; // Reset epoch count
    generate_weights(); // Reinitialize weights
}
// Function to reset previous weight updates and gradients
void reset_weight_updates()
{
    for (int i = 0; i < 9; i++)
    {
        prev_weight_update[i] = 0; // Reset previous weight updates
        update_weights[i] = 0; // Reset current updates
        gradients[i] = 0; // Reset gradients
    }
}
// Function to reset errors
void reset_errors()
{
    for (int i = 0; i < 4; i++)
    {
        error[i] = 0; // Reset errors
    }
}
// Function to reset RMSE history
void reset_rmse_history()
{
    for (int i = 0; i < epoch; i++)
    {
        RMSE_array_error[i] = 0; // Reset RMSE history
    }
}
void train_neural_net()
{
    // Train the neural network for a specified number of epochs
    while (epoch < 20000)
    {
        train_for_epoch();
        // Calculate RMSE for this epoch and store it
        calc_RMSE_ERROR();
        RMSE_array_error[epoch] = RMSE_ERROR; // Store RMSE in array for analysis
        cout << "Epoch: " << epoch << endl; // Display current epoch
        epoch++; // Increment epoch count
        // Check for convergence and restart training if necessary
        if (should_restart_training())
        {
            restart_training();
        }
    }
}
float sigmoid_function(float x)
{
    // Sigmoid activation function
    return 1 / (1 + exp(-x));
}
void generate_weights()
{
    // Generate random initial weights for the neural network
    srand(time(NULL)); // Seed random number generator
    for (int i = 0; i < 9; i++)
    {
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Random values between -1.0 and 1.0
        cout << "Weight " << i << " = " << weights[i] << endl; // Display initialized weight
    }
    cout << "" << endl; // Print a blank line
}
void calc_hidden_layers(int x)
{
    // Calculate outputs of the hidden layers for a given input
    float input1 = training_data[x][0];
    float input2 = training_data[x][1];
    sum_h1 = (input1 * weights[0]) + (input2 * weights[2]) + (bias * weights[4]);
    sum_h2 = (input1 * weights[1]) + (input2 * weights[3]) + (bias * weights[5]);
    h1 = sigmoid_function(sum_h1); // Activation for first hidden neuron
    h2 = sigmoid_function(sum_h2); // Activation for second hidden neuron
}
void calc_output_neuron()
{
    // Calculate the output neuron value
    float weighted_sum = (h1 * weights[6]) + (h2 * weights[7]) + (bias * weights[8]);
    output_neuron = sigmoid_function(weighted_sum); // Activation for output neuron
}
void calc_error(int x)
{
    // Calculate the error for the output neuron
    error[x] = output_neuron - answer_data[x]; 
}
void calc_derivatives(int x)
{
    // Calculate derivatives for backpropagation
    float exp_sum_output = exp(sum_output);
    float exp_sum_h1 = exp(sum_h1);
    float exp_sum_h2 = exp(sum_h2);   
    float denom_output = pow(1 + exp_sum_output, 2);
    float denom_h1 = pow(1 + exp_sum_h1, 2);
    float denom_h2 = pow(1 + exp_sum_h2, 2);
    derivative_O1 = -error[x] * (exp_sum_output / denom_output); // Derivative for output
    derivative_h1 = (exp_sum_h1 / denom_h1) * weights[6] * derivative_O1; // Derivative for first hidden neuron
    derivative_h2 = (exp_sum_h2 / denom_h2) * weights[7] * derivative_O1; // Derivative for second hidden neuron
}
void calc_gradient(int x)
{
    // Calculate gradients for weight updates based on derivatives
    float input1_sigmoid = sigmoid_function(training_data[x][0]);
    float input2_sigmoid = sigmoid_function(training_data[x][1]);
    float bias_sigmoid = sigmoid_function(bias);
    gradients[0] = input1_sigmoid * derivative_h1; // Gradient for weight 0
    gradients[1] = input1_sigmoid * derivative_h2; // Gradient for weight 1
    gradients[2] = input2_sigmoid * derivative_h1; // Gradient for weight 2
    gradients[3] = input2_sigmoid * derivative_h2; // Gradient for weight 3
    gradients[4] = bias_sigmoid * derivative_h1;    // Gradient for weight 4 (bias)
    gradients[5] = bias_sigmoid * derivative_h2;    // Gradient for weight 5 (bias)
    gradients[6] = h1 * derivative_O1;               // Gradient for weight 6
    gradients[7] = h2 * derivative_O1;               // Gradient for weight 7
    gradients[8] = bias_sigmoid * derivative_O1;    // Gradient for weight 8 (bias)
}
void calc_updates()
{
    // Calculate weight updates using gradients and momentum
    for (int i = 0; i < 9; i++)
    {
        float gradient_update = learning_rate * gradients[i]; // Calculate the gradient update
        float momentum_update = momentum * prev_weight_update[i]; // Calculate the momentum update
        
        update_weights[i] = gradient_update + momentum_update; // Combine updates
        prev_weight_update[i] = update_weights[i]; // Store current update for momentum
    }
}
void update_new_weights()
{
    // Apply calculated updates to the weights
    for (int i = 0; i < 9; i++)
    {
        weights[i] += update_weights[i]; // Increment each weight by its update
    }
}
void calc_RMSE_ERROR()
{
    // Calculate the Root Mean Square Error (RMSE) for the current epoch
    const int num_errors = 4; // Number of errors
    float sum_squared_errors = 0.0f;
    for (int i = 0; i < num_errors; i++)
    {
        sum_squared_errors += pow(error[i], 2); // Sum of squared errors
    }
    RMSE_ERROR = sqrt(sum_squared_errors / num_errors); // Calculate RMSE
    cout << "RMSE error: " << fixed << setprecision(4) << RMSE_ERROR << endl; // Display RMSE
}
