import numpy as np
import math

class XORModel:
    def __init__(self, input_size, hidden_size, output_size, seed=1234):
        """
        Initialize the XOR Model.

        :param input_size: Number of input neurons
        :param hidden_size: Number of hidden neurons
        :param output_size: Number of output neurons
        :param seed: Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = self._initialize_weights(hidden_size, input_size, seed)
        self.hidden_biases = np.zeros(hidden_size)
        self.output_weights = self._initialize_weights(output_size, hidden_size, seed)
        self.output_biases = np.zeros(output_size)
        self.hidden_outputs = np.zeros(hidden_size)

    def _initialize_weights(self, rows, cols, seed):
        """
        Initialize weights with a uniform distribution between -0.5 and 0.5.

        :param rows: Number of rows
        :param cols: Number of columns
        :param seed: Random seed
        :return: Initialized weights
        """
        np.random.seed(seed)
        return np.random.uniform(-0.5, 0.5, size=(rows, cols))

    def activation(self, x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + math.exp(-x))

    def hidden_activation(self, x):
        """ReLU activation function for hidden layer."""
        return max(0, x)

    def activation_derivative(self, x):
        """Derivative of sigmoid activation function."""
        return x * (1.0 - x)

    def calculate_error(self, output, target):
        """Mean Squared Error (MSE)."""
        return 0.5 * (output - target) ** 2

    def forward(self, input_vector):
        """
        Forward pass through the network.

        :param input_vector: Input vector
        :return: Output of the network
        """
        # Hidden layer
        for i in range(self.hidden_size):
            sum = self.hidden_biases[i]
            for j in range(self.input_size):
                sum += self.hidden_weights[i, j] * input_vector[j]
            self.hidden_outputs[i] = self.hidden_activation(sum)

        # Output layer
        output = self.output_biases[0]
        for j in range(self.hidden_size):
            output += self.output_weights[0, j] * self.hidden_outputs[j]
        output = self.activation(output)
        return output

    def train(self, inputs, targets, epochs, learning_rate):
        """
        Train the network using backpropagation.

        :param inputs: Input vectors
        :param targets: Target outputs
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for weight updates
        """
        errors = []
        for epoch in range(epochs):
            total_error = 0.0
            for i in range(len(inputs)):
                output = self.forward(inputs[i])
                total_error += self.calculate_error(output, targets[i])
                self.backpropagate(inputs[i], targets[i], output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}: Total error = {total_error}")
                errors.append(total_error)
        # Write errors to a file
        with open("errors.txt", "w") as f:
            for error in errors:
                f.write(str(error) + "\n")

    def compute_accuracy(self, inputs, targets):
        """
        Compute the accuracy of the network.

        :param inputs: Input vectors
        :param targets: Target outputs
        :return: Accuracy as a percentage
        """
        correct_predictions = 0
        for i in range(len(inputs)):
            output = self.forward(inputs[i])
            predicted = 1 if output >= 0.5 else 0
            actual = int(targets[i])
            correct_predictions += (predicted == actual)
        return (correct_predictions / len(inputs)) * 100.0

    def backpropagate(self, input_vector, target, output, learning_rate):
        """
        Backward pass to update weights and biases.

        :param input_vector: Input vector
        :param target: Target output
        :param output: Actual output
        :param learning_rate: Learning rate for weight updates
        """
        # Output layer error and gradient
        output_error = (output - target) * self.activation_derivative(output)

        # Update output layer weights and biases
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.output_weights[i, j] -= learning_rate * output_error * self.hidden_outputs[j]
            self.output_biases[i] -= learning_rate * output_error

        # Hidden layer error and gradient
        for i in range(self.hidden_size):
            hidden_error = 0.0
            for j in range(self.output_size):
                hidden_error += output_error * self.output_weights[j, i]
            hidden_error *= 1 if self.hidden_outputs[i] > 0 else 0  # Derivative of ReLU

            # Update hidden layer weights and biases
            for j in range(self.input_size):
                self.hidden_weights[i, j] -= learning_rate * hidden_error * input_vector[j]
            self.hidden_biases[i] -= learning_rate * hidden_error


# Example usage
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    learning_rates = [0.02]
    hidden_layer_sizes = [20, 50, 100]

    for learning_rate in learning_rates:
        for hidden_layer_size in hidden_layer_sizes:
            model = XORModel(2, hidden_layer_size, 1)
            model.train(X, y, 10000, learning_rate)

            # Evaluate the model
            for i in range(len(X)):
                output = model.forward(X[i])
                print(f"Input: {X[i]} => Predicted Output: {output}, Actual Output: {y[i]}")

            # Calculate and print accuracy
            accuracy = model.compute_accuracy(X, y)
            print(f"Accuracy: {accuracy}%")
