import numpy as np
from visualize import NeuralNetworkVisualizer


class NeuralNetwork:
    def __init__(self, layers, activation="relu", visualize=True):
        self.layers = layers
        self.activation = activation
        self.visualize = visualize
        self.weights = []
        self.biases = []

        if self.visualize:
            global visualizer
            visualizer = NeuralNetworkVisualizer()

        # Initialize weights and biases
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward(self, X):
        activations = [X]
        inputs = []

        for i in range(len(self.weights)):
            input_to_layer = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            inputs.append(input_to_layer)

            if self.activation == "relu":
                activations.append(self.relu(input_to_layer))
            elif self.activation == "sigmoid":
                activations.append(self.sigmoid(input_to_layer))

        return activations, inputs

    def backward(self, X, y, activations, inputs):
        m = y.shape[0]
        d_weights = [None] * len(self.weights)
        d_biases = [None] * len(self.biases)

        error = activations[-1] - y
        delta = error

        for i in reversed(range(len(self.weights))):
            if self.activation == "relu":
                delta *= self.relu_derivative(inputs[i])
            elif self.activation == "sigmoid":
                delta *= self.sigmoid_derivative(inputs[i])

            d_weights[i] = np.dot(activations[i].T, delta) / m
            d_biases[i] = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)

        return d_weights, d_biases

    def update(self, d_weights, d_biases, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * d_weights[i]
            self.biases[i] -= learning_rate * d_biases[i]

    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            activations, inputs = self.forward(X)
            d_weights, d_biases = self.backward(X, y, activations, inputs)
            self.update(d_weights, d_biases, learning_rate)

            loss = np.mean(np.square(activations[-1] - y))
            losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                if self.visualize:
                    visualizer.update_loss_plot(epoch, loss)
        if self.visualize:
            visualizer.finalize_plot()

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def visualize_predictions(self, X, y):
        predictions = self.predict(X)
        NeuralNetworkVisualizer.plot_predictions(y, predictions)
