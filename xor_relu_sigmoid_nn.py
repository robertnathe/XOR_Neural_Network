import numpy as np
from typing import Tuple, Optional

class XORModel:
    """
    A simple feed-forward neural network with one hidden layer designed to solve the XOR problem.
    Uses ReLU for the hidden layer and Sigmoid for the output layer.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: Optional[int] = None):
        """
        Initializes the XORModel with specified layer sizes.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of neurons in the output layer.
            seed (Optional[int]): Seed for the random number generator, ensuring reproducible weight initialization.

        Raises:
            ValueError: If any size parameter is not a positive integer.
        """
        # 🧱 Initialization
        # Validate input parameters
        if not all(isinstance(s, int) and s > 0 for s in [input_size, hidden_size, output_size]):
            raise ValueError("Input, hidden, and output sizes must be positive integers.")

        # Store sizes for clarity and consistent validation
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        # Use np.random.default_rng() for modern random number generation.
        # Passing the 'seed' parameter directly ensures reproducibility of weight initialization
        # for this specific model instance, independent of any global np.random.seed().
        rng = np.random.default_rng(seed)

        # Weights: initialized with small random numbers to break symmetry and aid learning.
        # Biases: initialized to zero.
        # Explicitly setting dtype to np.float64 for numerical consistency across the model,
        # preventing potential precision issues.
        # Using a consistent random range for initial weights.
        self.hidden_weights: np.ndarray = rng.uniform(-0.5, 0.5, (input_size, hidden_size)).astype(np.float64)
        self.output_weights: np.ndarray = rng.uniform(-0.5, 0.5, (hidden_size, output_size)).astype(np.float64)
        
        self.hidden_biases: np.ndarray = np.zeros((1, hidden_size), dtype=np.float64)
        self.output_biases: np.ndarray = np.zeros((1, output_size), dtype=np.float64)

    # ⚙️ Activation Functions
    def hidden_activation(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU (Rectified Linear Unit) activation function.
        Args:
            x (np.ndarray): Input array to the activation function.
        Returns:
            np.ndarray: Output array with ReLU applied (max(0, x)).
        """
        return np.maximum(0, x)

    def hidden_activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the ReLU activation function.
        Args:
            x (np.ndarray): Input array from the pre-activation of the hidden layer (z_h).
        Returns:
            np.ndarray: Derivative values (1 where x > 0, 0 otherwise).
                        Ensures output type is float64 for consistency in calculations.
        """
        return (x > 0).astype(np.float64)

    def activation(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        Args:
            x (np.ndarray): Input array to the activation function.
        Returns:
            np.ndarray: Output array with Sigmoid applied (1 / (1 + exp(-x))).
        """
        # For robustness against extremely large negative x values, one could use a more numerically
        # stable implementation, but for typical neural network values, the standard formula is fine.
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, output: np.ndarray) -> np.ndarray:
        """
        Derivative of the Sigmoid function.
        Args:
            output (np.ndarray): The *already activated* output array from the Sigmoid function itself (a_o).
        Returns:
            np.ndarray: Derivative values (output * (1 - output)).
        """
        return output * (1 - output)

    # ➡️ Forward Pass
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the output of the network for a given input batch and returns intermediate values
        required for backpropagation.
        
        Args:
            X (np.ndarray): Input data, a 2D array where rows are samples and columns are features.
                            Expected shape: (num_samples, input_size).
        
        Returns:
            Tuple[predicted_output, hidden_layer_output, hidden_layer_input]:
            - predicted_output (np.ndarray): The final output of the network (e.g., probabilities for XOR),
                                             shape: (num_samples, output_size).
            - hidden_layer_output (np.ndarray): The activated output of the hidden layer (a_h),
                                                shape: (num_samples, hidden_size).
            - hidden_layer_input (np.ndarray): The linear input to the hidden layer before activation (z_h),
                                               shape: (num_samples, hidden_size).
        
        Raises:
            TypeError: If X is not a numpy array.
            ValueError: If X has incorrect dimensions or feature count.
            RuntimeError: For unexpected errors during computation, chaining the original exception.
        """
        # Input validation for robustness and clear error messages.
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError(f"Input X must be a 2D array (samples, features), got {X.ndim}D.")
        if X.shape[1] != self.input_size:
            raise ValueError(f"Input X feature dimension ({X.shape[1]}) must match model input size ({self.input_size}).")

        try:
            # Hidden Layer computation:
            # Linear combination: input_data @ weights + biases.
            # Use the @ operator for highly optimized matrix multiplication in Python 3.5+.
            hidden_layer_input: np.ndarray = X @ self.hidden_weights + self.hidden_biases
            hidden_layer_output: np.ndarray = self.hidden_activation(hidden_layer_input)
            
            # Output Layer computation:
            output_layer_input: np.ndarray = hidden_layer_output @ self.output_weights + self.output_biases
            predicted_output: np.ndarray = self.activation(output_layer_input)
            
            # Return predicted output and intermediate values needed for backpropagation.
            return predicted_output, hidden_layer_output, hidden_layer_input
        except ValueError as ve:
            # Catch specific shape mismatch errors from matrix multiplication and chain the exception for better debugging.
            raise ValueError(f"Shape mismatch error during forward pass: {ve}") from ve
        except Exception as e:
            # Catch any other unexpected errors during computation and chain the exception.
            raise RuntimeError(f"An unexpected error occurred during forward pass: {e}") from e

    # ↩️ Backpropagation
    def backpropagate(self, X: np.ndarray, y: np.ndarray, predicted_output: np.ndarray, 
                      hidden_layer_output: np.ndarray, hidden_layer_input: np.ndarray, 
                      learning_rate: float) -> None:
        """
        Implements the backpropagation algorithm to adjust weights and biases using batch gradient descent.
        Accepts intermediate layer outputs from the forward pass as arguments to avoid recomputation.

        Args:
            X (np.ndarray): Input data used for the forward pass, shape: (num_samples, input_size).
            y (np.ndarray): True target labels corresponding to X, shape: (num_samples, output_size).
            predicted_output (np.ndarray): The network's output from the forward pass (a_o),
                                           shape: (num_samples, output_size).
            hidden_layer_output (np.ndarray): Activated output of the hidden layer from forward pass (a_h),
                                              shape: (num_samples, hidden_size).
            hidden_layer_input (np.ndarray): Linear input to the hidden layer from forward pass (z_h),
                                             shape: (num_samples, hidden_size).
            learning_rate (float): The step size for updating weights and biases.
        
        Raises:
            TypeError: If X, y, predicted_output, hidden_layer_output, or hidden_layer_input are not numpy arrays.
            ValueError: If learning_rate is not a positive number or if shapes of inputs are incompatible.
            RuntimeError: For unexpected errors during computation, chaining the original exception.
        """
        # Input validation for robustness.
        if not all(isinstance(arr, np.ndarray) for arr in [X, y, predicted_output, hidden_layer_output, hidden_layer_input]):
            raise TypeError("All input arrays (X, y, predicted_output, hidden_layer_output, hidden_layer_input) must be numpy arrays.")
        # Added np.floating to robustly handle NumPy scalar types for learning_rate.
        if not isinstance(learning_rate, (int, float, np.floating)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number.")

        # Ensure shapes are compatible before starting computation, adding robustness.
        num_samples = X.shape[0]
        if not (num_samples == y.shape[0] == predicted_output.shape[0] == hidden_layer_output.shape[0] == hidden_layer_input.shape[0]):
            raise ValueError("Number of samples across all input arrays (X, y, predicted_output, intermediate layers) must be the same.")
        if y.shape[1] != predicted_output.shape[1] or y.shape[1] != self.output_size:
            raise ValueError(f"Output dimension of y ({y.shape[1]}) and predicted_output ({predicted_output.shape[1]}) must be the same and match model output size ({self.output_size}).")
        if hidden_layer_output.shape[1] != self.hidden_size or hidden_layer_input.shape[1] != self.hidden_size:
            raise ValueError(f"Hidden layer dimensions ({hidden_layer_output.shape[1]}, {hidden_layer_input.shape[1]}) must match model hidden size ({self.hidden_size}).")

        try:
            # 1. Output Layer Error and Delta (using Mean Squared Error derivative: dL/da_o = (a_o - y))
            output_error: np.ndarray = predicted_output - y
            # Delta for output layer (dL/dz_o = dL/da_o * da_o/dz_o)
            output_delta: np.ndarray = output_error * self.activation_derivative(predicted_output)
            
            # 2. Hidden Layer Error and Delta
            # Error propagated back from the output layer to the hidden layer (dL/da_h).
            # This is (dL/dz_o) @ W_output.T
            hidden_error: np.ndarray = output_delta @ self.output_weights.T
            # Delta for hidden layer (dL/dz_h = dL/da_h * da_h/dz_h)
            hidden_delta: np.ndarray = hidden_error * self.hidden_activation_derivative(hidden_layer_input)

            # 3. Update Weights and Biases (Gradient Descent)
            # Gradients are calculated by summing over the batch dimension (dL/dW = a_h.T @ dL/dz_o and X.T @ dL/dz_h).
            # Updates are performed in-place.
            self.output_weights -= learning_rate * (hidden_layer_output.T @ output_delta)
            # Biases gradients are summed across samples for their gradient (dL/db = sum(dL/dz) over samples).
            self.output_biases -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
            
            self.hidden_weights -= learning_rate * (X.T @ hidden_delta)
            self.hidden_biases -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        except ValueError as ve:
            # Catch specific shape mismatch errors or numerical issues and chain the exception.
            raise ValueError(f"Shape mismatch or numerical error during backpropagation: {ve}") from ve
        except Exception as e:
            # Catch any other unexpected errors and chain the exception.
            raise RuntimeError(f"An unexpected error occurred during backpropagation: {e}") from e

    # ✅ Prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates binary predictions (0 or 1) for the given input data batch.
        The output layer's sigmoid activation is thresholded at 0.5.

        Args:
            X (np.ndarray): Input data to predict, shape: (num_samples, input_size).
                            Validation for X (type, dimensions, feature count) is performed
                            within the `forward` method this function calls.

        Returns:
            np.ndarray: Binary predictions (0.0 or 1.0), shape: (num_samples, output_size),
                        with dtype np.float64 for consistency with target 'y' typically being float.

        Raises:
            RuntimeError: For unexpected errors during prediction, chaining the original exception.
                          Includes errors originating from `self.forward(X)`'s validation.
        """
        try:
            # self.forward(X) performs all necessary input validation for X (type, dimensions, feature count).
            predicted_output, _, _ = self.forward(X) 
            # Convert continuous output to binary prediction (0.0 if < 0.5, 1.0 if >= 0.5).
            # Using np.float64 for the output type ensures consistency with typical target y dtypes.
            predictions: np.ndarray = (predicted_output >= 0.5).astype(np.float64)
            return predictions
        except Exception as e:
            # Chain the exception for clearer traceback, including validation errors from forward().
            raise RuntimeError(f"Error during prediction: {e}") from e

    # ✅ Accuracy
    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the percentage of correct predictions for the given input data.
        Utilizes the `predict` method for generating binary outcomes.

        Args:
            X (np.ndarray): Input data to evaluate, shape: (num_samples, input_size).
                            Validation for X (type, dimensions, feature count) is performed
                            within the `predict` method this function calls.
            y (np.ndarray): True target labels for X, shape: (num_samples, output_size).

        Returns:
            float: The accuracy as a percentage (0.0 to 100.0).
        
        Raises:
            TypeError: If y is not a numpy array.
            ValueError: If shapes are incompatible (e.g., X and y sample counts differ,
                        or y output dimension does not match model output size).
            RuntimeError: For unexpected errors during accuracy computation, chaining the original exception.
                          Includes errors originating from `self.predict(X)`'s validation.
        """
        # Validate 'y' and its compatibility with the model's output and predictions from X.
        if not isinstance(y, np.ndarray):
            raise TypeError("Target y must be a numpy array.")

        try:
            # self.predict(X) performs all necessary input validation for X.
            predictions: np.ndarray = self.predict(X) 
            
            # Now validate 'y' against the returned predictions (and implicitly X's sample count)
            if y.shape[0] != predictions.shape[0]: 
                raise ValueError(f"Number of samples in target y ({y.shape[0]}) must match number of predictions ({predictions.shape[0]}).")
            if y.shape[1] != self.output_size:
                raise ValueError(f"Target y output dimension ({y.shape[1]}) must match model output size ({self.output_size}).")
            
            # Calculate accuracy.
            correct_predictions_count: int = np.sum(predictions == y)
            total_samples: int = y.shape[0] 
            accuracy: float = (correct_predictions_count / total_samples) * 100
            
            return accuracy
        except Exception as e:
            # Chain the exception for clearer traceback.
            raise RuntimeError(f"Error calculating accuracy: {e}") from e

    # 🏋️ Training
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> None:
        """
        Trains the model for a specified number of epochs using batch gradient descent.

        Args:
            X (np.ndarray): Training input data, shape: (num_samples, input_size).
            y (np.ndarray): Training target labels, shape: (num_samples, output_size).
            epochs (int): The number of training iterations. Must be a positive integer.
            learning_rate (float): The learning rate for weight and bias updates. Must be a positive number.
        
        Raises:
            TypeError: If X or y are not numpy arrays.
            ValueError: If shapes are incompatible, epochs not positive, or learning_rate not positive.
            Exception: Re-raises any exceptions encountered during training after logging the epoch.
        """
        # Validate inputs for training.
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Input X and target y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")
        # X.shape[1] validation is handled by self.forward(X) which is called inside the loop.
        if y.shape[1] != self.output_size:
            raise ValueError(f"Target y output dimension ({y.shape[1]}) must match model output size ({self.output_size}).")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")
        if not isinstance(learning_rate, (int, float, np.floating)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number.")

        print("--- Starting Training ---")
        epoch: int = 0  # Initialize epoch outside the loop for consistent error reporting in finally/except.
        try:
            for epoch in range(epochs):
                # Forward pass: unpack all necessary values from the forward pass for backpropagation.
                predicted_output, hidden_layer_output, hidden_layer_input = self.forward(X)
                
                # Backpropagation: pass all necessary values to backpropagate.
                self.backpropagate(X, y, predicted_output, hidden_layer_output, hidden_layer_input, learning_rate)
                
                # Monitoring (calculate and print Mean Squared Error)
                # Print every 100 epochs or on the very last epoch to show progress without excessive output.
                if (epoch + 1) % 100 == 0 or epoch == epochs - 1: 
                    error: float = np.mean(np.square(y - predicted_output))
                    print(f"Epoch {epoch + 1}/{epochs}: Error = {error:.6f}")
        except Exception as e:
            # Log the specific epoch where the error occurred and re-raise the exception.
            print(f"An error occurred during training at epoch {epoch + 1}: {e}")
            raise # Re-raise the exception after logging.
        finally:
            print("--- Training Attempt Complete ---")


# 🚀 Execution
if __name__ == '__main__':
    try:
        # Data Definition for the XOR problem.
        # Explicitly define dtype as np.float64 for consistency with model weights/biases.
        X: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        # Target outputs must be in the correct shape (num_samples, output_size), i.e., (4, 1).
        y: np.ndarray = np.array([[0], [1], [1], [0]], dtype=np.float64) 

        # Model Instantiation
        input_size: int = 2
        hidden_size: int = 4 # A common choice for XOR, allows the model to learn non-linearity.
        output_size: int = 1
        xor_model: XORModel = XORModel(input_size, hidden_size, output_size, seed=42) # Pass seed for reproducibility

        # Training
        epochs: int = 10000
        learning_rate: float = 0.1
        xor_model.train(X, y, epochs, learning_rate)
        print("--- Training Complete ---")

        # Evaluation
        final_accuracy: float = xor_model.compute_accuracy(X, y)
        print(f"\nFinal Accuracy: {final_accuracy:.2f}%")

        # Prediction for individual samples
        print("\n--- Final Predictions ---")
        # Perform forward pass once for the entire batch to get raw sigmoid outputs for printing.
        all_predicted_outputs, _, _ = xor_model.forward(X)
        
        for i, (input_data, target) in enumerate(zip(X, y)):
            # Extract the raw predicted output (e.g., 0.98) for the current sample.
            prediction_raw: float = all_predicted_outputs[i][0]
            # Convert raw output to binary prediction (0 or 1).
            binary_prediction: int = 1 if prediction_raw >= 0.5 else 0
            # Cast target to int for clean output display.
            print(f"Input: {input_data}, Target: {int(target[0])}, Predicted Output: {prediction_raw:.4f}, Final Prediction: {binary_prediction}")

    except Exception as e:
        # Catch and log any exceptions that occur during the global execution flow.
        print(f"\nAn error occurred during global execution: {e}")
