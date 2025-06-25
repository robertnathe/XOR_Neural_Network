import math
import random

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

class XORModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases with He/Xavier initialization
        random.seed(1234)
        
        # He initialization for ReLU hidden layer
        stddev_h = math.sqrt(2.0 / input_size)
        self.hidden_weights = [
            [random.gauss(0, stddev_h) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.hidden_biases = [0.0] * hidden_size
        
        # Xavier initialization for sigmoid output layer
        stddev_o = math.sqrt(1.0 / hidden_size)
        self.output_weights = [
            [random.gauss(0, stddev_o) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]
        self.output_biases = [0.0] * output_size
        
        # Hidden layer outputs cache
        self.hidden_outputs = [0.0] * hidden_size

    def forward(self, input_data):
        # Hidden layer computation
        for i in range(self.hidden_size):
            total = self.hidden_biases[i]
            for j in range(self.input_size):
                total += self.hidden_weights[i][j] * input_data[j]
            self.hidden_outputs[i] = self.hidden_activation(total)
        
        # Output layer computation
        output_sum = self.output_biases[0]
        for j in range(self.hidden_size):
            output_sum += self.output_weights[0][j] * self.hidden_outputs[j]
        return self.activation(output_sum)

    def train(self, inputs, targets, epochs, learning_rate):
        errors = []
        for epoch in range(epochs):
            total_error = 0.0
            for i in range(4):
                output = self.forward(inputs[i])
                total_error += self.calculate_error(output, targets[i])
                self.backpropagate(inputs[i], targets[i], output, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Error = {total_error}")
                errors.append(total_error)
    
    def compute_accuracy(self):
        correct = 0
        for i in range(4):
            output = self.forward(X[i])
            if (output >= 0.5) == (y[i] >= 0.5):
                correct += 1
        return (correct / 4.0) * 100.0

    # Activation functions
    def activation(self, x):  # Sigmoid
        return 1.0 / (1.0 + math.exp(-x))
    
    def hidden_activation(self, x):  # ReLU
        return x if x > 0 else 0
    
    def activation_derivative(self, x):  # Sigmoid derivative
        return x * (1.0 - x)
    
    def relu_derivative(self, x):
        return 1.0 if x > 0 else 0.0
    
    def calculate_error(self, output, target):
        return 0.5 * (output - target) ** 2
    
    def backpropagate(self, input_data, target, output, learning_rate):
        # Output layer error
        output_error = (output - target) * self.activation_derivative(output)
        
        # Update output weights and biases
        for j in range(self.hidden_size):
            self.output_weights[0][j] -= learning_rate * output_error * self.hidden_outputs[j]
        self.output_biases[0] -= learning_rate * output_error
        
        # Hidden layer error
        for i in range(self.hidden_size):
            # Calculate error contribution from this hidden neuron
            hidden_error = output_error * self.output_weights[0][i]
            # Apply ReLU derivative
            hidden_error *= self.relu_derivative(self.hidden_outputs[i])
            
            # Update hidden weights and biases
            for j in range(self.input_size):
                self.hidden_weights[i][j] -= learning_rate * hidden_error * input_data[j]
            self.hidden_biases[i] -= learning_rate * hidden_error

if __name__ == "__main__":
    # Experiment with parameters
    learning_rates = [0.02]
    hidden_sizes = [20, 50, 100]
    
    for lr in learning_rates:
        for hs in hidden_sizes:
            print(f"\nTraining with lr={lr}, hidden_size={hs}")
            model = XORModel(2, hs, 1)
            model.train(X, y, 3000, lr)
            
            # Test predictions
            for i in range(4):
                pred = model.forward(X[i])
                print(f"{X[i][0]} XOR {X[i][1]} = {pred:.6f} (expected {y[i]})")
            
            accuracy = model.compute_accuracy()
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Parameters: learning_rate={lr}, hidden_size={hs}")
