import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize parameters
def initialize_parameters(input_dim, hidden_dim, output_dim):
    np.random.seed(42)  # For reproducibility
    W1 = np.random.randn(input_dim, hidden_dim)  # Weights for input to hidden layer
    b1 = np.zeros((1, hidden_dim))  # Biases for hidden layer
    W2 = np.random.randn(hidden_dim, output_dim)  # Weights for hidden to output layer
    b2 = np.zeros((1, output_dim))  # Biases for output layer
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2, A1

# Compute cost (mean squared error)
def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = np.sum((A2 - Y) ** 2) / (2 * m)
    return cost

# Backward propagation
def backward_propagation(X, Y, A2, A1, W2):
    m = X.shape[0]
    dA2 = A2 - Y
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

# Update parameters using gradient descent
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Training the neural network
def train(X, Y, hidden_dim, epochs, learning_rate):
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    W1, b1, W2, b2 = initialize_parameters(input_dim, hidden_dim, output_dim)
    
    for epoch in range(epochs):
        A2, A1 = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A2, A1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost {cost}")

    return W1, b1, W2, b2

# Predict function
def predict(X, W1, b1, W2, b2):
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    return A2

# Sample data (for demonstration purposes)
X = np.array([
    [0.1, 0.2],
    [0.9, 0.8],
    [0.8, 0.9],
    [0.2, 0.1]
])
Y = np.array([
    [0.1],
    [0.9],
    [0.8],
    [0.2]
])

# Training parameters
hidden_dim = 5
epochs = 1000
learning_rate = 0.01

# Train the model
W1, b1, W2, b2 = train(X, Y, hidden_dim, epochs, learning_rate)

# Predict
predictions = predict(X, W1, b1, W2, b2)

print("Predicted Output after training:")
print(predictions)          
