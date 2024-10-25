import matplotlib.pyplot as plt

# Read the error data from the file
with open('errors.txt', 'r') as f:
    errors = [float(line.strip()) for line in f.readlines()]

# Plot the error over epochs
plt.plot(errors)
plt.title('Total Error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Total Error')
plt.grid()

# Save the plot as an image
plt.savefig('error_plot.png')  # Saves the plot as a PNG file
plt.close()  # Close the plot to free up memory
