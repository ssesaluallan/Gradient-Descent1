import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
np.random.seed(3) # Ensure random numbers are the same each time the code is run

# Generate the dataset of 450 2D points
X, y = make_blobs(n_samples=450, centers=3, cluster_std=1.5)
K = 3 # Output classes
D = X.shape[1] # 2D input features
H = 32 # Number of neurons in the hidden layer


# W2 and W2 for weight initialisation using random normal distribution scaled by the size of the incoming layer
# b1 and b2 for bias vectors initialised to zero
W1 = np.random.randn(D, H) / np.sqrt(2/D)
b1 = np.zeros(H)
W2 = np.random.randn(H, K) * np.sqrt(2 / H)
b2 = np.zeros(K)

# Training settings
lr = 0.05 # Learning rate
epochs = 200 # Number of times the training loop will run
batch = 64 # Batch sizes(mini batch size)
losses = []

# Activation functions(relu and softmax)
relu = lambda z: np.maximum(0, z) # Replaces neg numbers in matrix with zero for non-linearity to the network

# Convert raw output scores to probability distribution
# Where all values are positive and add upto 1
def softmax(z):
    z -= z.max(axis=1, keepdims=True) # Prevent overflow errors
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)

# Training loop
for _ in range(epochs):
    idx = np.random.permutation(len(X)) # Randomly shuffles the data indices
    Xs, Ys = X[idx], y[idx]
    for i in range(0, len(Xs), batch):
        # Mini batching(Loop through data in chunks of 64 samples at a time)
        xb = Xs[i:i + batch]
        yb = Ys[i:i + batch]

        # Forward pass 
        z1 = xb.dot(W1) + b1
        a1 = relu(z1) # Applies relu to get hidden features
        logits = a1.dot(W2) + b2 # Gets raw class scores
        probs = softmax(logits) # Pass through softmax to get predictions

        # Loss calculation
        Yb = np.zeros_like(probs)

        # Backward pass(Gradients) 
        # Uses chain rule to calculate how much the loss would change if any weight or bias changed
        Yb[np.arange(len(yb)), yb] = 1
        dlog = (probs - Yb) / len(yb)
        dW2 = a1.T.dot(dlog)
        db2 = dlog.sum(axis=0)
        da1 = dlog.dot(W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = xb.T.dot(dz1)
        db1 = dz1.sum(axis=0)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    # Optimization(Gradient descent)
    # Updates weights and biases by subtracting learning rate(lr) of the gradients to lower the loss
    logits_full = relu(X.dot(W1) + b1).dot(W2) + b2
    pf = softmax(logits_full)
    Yf = np.zeros_like(pf)
    Yf[np.arange(len(y)), y] = 1
    losses.append(-(Yf * np.log(pf + 1e-12)).mean()) # Append Average Cross-Entropy Loss across dataset to track the training progress

# Visualization and performance
plt.figure()
plt.plot(losses, color="green")
plt.title("Neural Network — Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Decision boundary grid
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]
pred = np.argmax(softmax(relu(grid.dot(W1) + b1).dot(W2) + b2),
                 axis=1).reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, pred, K, cmap="Greens")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Greens", edgecolor="black")
plt.title("Neural Network — Decision Regions")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()