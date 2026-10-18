import numpy as np
import time

def sign_activation(x):
    return np.where(x >= 0, 1, -1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# XOR input and output
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y = np.array([[-1], [1], [1], [-1]])

# Initialize weights with better scaling
np.random.seed(0)  # For reproducibility
input_size = 2
hidden_size = 4  # Increased hidden layer size
output_size = 1

# He initialization for better gradient flow
weights_hidden = np.random.normal(0, np.sqrt(2.0/input_size), (hidden_size, input_size))
bias_hidden = np.zeros((hidden_size, 1))
weights_output = np.random.normal(0, np.sqrt(2.0/hidden_size), (output_size, hidden_size))
bias_output = np.zeros((output_size, 1))

# Previous weight updates for momentum
prev_update_hidden = np.zeros_like(weights_hidden)
prev_update_output = np.zeros_like(weights_output)
prev_update_bias_hidden = np.zeros_like(bias_hidden)
prev_update_bias_output = np.zeros_like(bias_output)

# Hyperparameters
learning_rate = 0.1
momentum = 0.9
epochs = 1000
batch_size = 2  # Mini-batch size
total_time = 0

print("Training started...")
for epoch in range(epochs):
    epoch_start = time.time()
    total_error = 0
    
    # Shuffle training data
    indices = np.random.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Mini-batch training
    for i in range(0, X.shape[0], batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        batch_error = 0
        
        # Accumulate gradients over mini-batch
        dW_hidden = np.zeros_like(weights_hidden)
        dW_output = np.zeros_like(weights_output)
        db_hidden = np.zeros_like(bias_hidden)
        db_output = np.zeros_like(bias_output)
        
        for j in range(len(batch_X)):
            # Forward pass with tanh activation
            hidden_input = np.dot(weights_hidden, batch_X[j].reshape(-1, 1)) + bias_hidden
            hidden_output = tanh(hidden_input)
            final_input = np.dot(weights_output, hidden_output) + bias_output
            final_output = sign_activation(final_input)
            
            # Error calculation
            error = batch_y[j].reshape(-1, 1) - final_output
            batch_error += np.abs(error)
            
            # Backpropagation
            delta_output = error
            delta_hidden = np.dot(weights_output.T, delta_output) * tanh_derivative(hidden_output)
            
            # Accumulate gradients
            dW_output += delta_output @ hidden_output.T
            db_output += delta_output
            dW_hidden += delta_hidden @ batch_X[j].reshape(1, -1)
            db_hidden += delta_hidden
        
        # Average gradients over mini-batch
        dW_output /= len(batch_X)
        db_output /= len(batch_X)
        dW_hidden /= len(batch_X)
        db_hidden /= len(batch_X)
        
        # Update weights with momentum
        update_output = learning_rate * dW_output + momentum * prev_update_output
        update_hidden = learning_rate * dW_hidden + momentum * prev_update_hidden
        update_bias_output = learning_rate * db_output + momentum * prev_update_bias_output
        update_bias_hidden = learning_rate * db_hidden + momentum * prev_update_bias_hidden
        
        weights_output += update_output
        weights_hidden += update_hidden
        bias_output += update_bias_output
        bias_hidden += update_bias_hidden
        
        # Store updates for next iteration
        prev_update_output = update_output
        prev_update_hidden = update_hidden
        prev_update_bias_output = update_bias_output
        prev_update_bias_hidden = update_bias_hidden
        
        total_error += batch_error
    
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    total_time += epoch_time
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Error: {total_error[0][0]:.4f}, Time: {epoch_time:.4f}s")

print(f"\nTraining completed in {total_time:.2f} seconds")

# Test
print("\nTesting Results:")
correct = 0
total = X.shape[0]

for i in range(X.shape[0]):
    hidden_input = np.dot(weights_hidden, X[i].reshape(-1, 1)) + bias_hidden
    hidden_output = tanh(hidden_input)
    final_input = np.dot(weights_output, hidden_output) + bias_output
    final_output = sign_activation(final_input)
    
    print(f"Input: {X[i]}, Target: {y[i][0]}, Output: {final_output[0][0]}")
    if y[i][0] == final_output[0][0]:
        correct += 1

accuracy = (correct / total) * 100
print(f"\nAccuracy: {accuracy:.1f}%") 