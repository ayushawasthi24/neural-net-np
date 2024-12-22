import numpy as np
from nn import NeuralNetwork

# Example Usage
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = np.array([[x[0] + x[1]] for x in X])  # Sum of inputs as targets


# Define and train neural network
nn = NeuralNetwork([2, 5, 1], activation="relu")
nn.train(X, y, epochs=4000, learning_rate=0.01)

nn.visualize_predictions(X, y)

# Predict
predictions = nn.predict(X[:5])
print("Inputs", X[:5])
print("Predictions", predictions)
print("True Values", y[:5])
print("Loss", np.mean(np.square(predictions - y[:5])))
