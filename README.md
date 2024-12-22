# Neural Network from Scratch

This project implements a basic neural network from scratch using Python and NumPy. The neural network supports forward and backward propagation, weight updates, and training using gradient descent. It also includes the option to visualize the training process and predictions.

## Features

- Customizable layers and activation functions (ReLU, Sigmoid)
- Forward and backward propagation
- Gradient descent optimization for training
- Loss visualization during training
- Prediction and visualization of model performance
- Supports multi-layer feedforward neural networks

## Requirements

- Python 3.x
- NumPy

### Install dependencies:

You can install the necessary dependencies using pip:

```bash
pip install numpy
```

## Project Structure

The project consists of the following main files:

- `nn.py`: The main neural network implementation with forward propagation, backpropagation, and training functionality.
- `visualize.py`: Visualization module (optional) to display loss and predictions during training.

## Usage

### 1. Initialize the Neural Network

You can initialize the neural network with the desired number of layers and activation function.

```python
from nn import NeuralNetwork

# Example: Neural network with 2 input nodes, 3 hidden nodes, and 1 output node using ReLU activation
nn = NeuralNetwork(layers=[2, 3, 1], activation="relu", visualize=True)
```

### 2. Train the Network

Train the neural network by passing in training data, labels, number of epochs, and learning rate.

```python
# Example training data (X) and labels (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the network with 1000 epochs and a learning rate of 0.1
nn.train(X, y, epochs=1000, learning_rate=0.1)
```

### 3. Make Predictions

Once the network is trained, you can make predictions on new data:

```python
predictions = nn.predict(X)
print("Predictions:", predictions)
```

### 4. Visualize Predictions

If you have the `visualize` module, you can visualize the predictions versus the true labels:

```python
nn.visualize_predictions(X, y)
```

### 5. Loss Visualization

During training, the loss will be plotted automatically if `visualize=True` is set in the network initialization. The plot will show the loss over time, helping to monitor the progress of training.

### 6. Sample usage

Sample usage is provided in the `main.py` file.

```python
python main.py
```

This file creates a random 100 sample dataset with 2 features and the output vector as the sum of features for each sample. And training is done on this random dataset.

## Math Behind the Network

This neural network follows the basic principles of forward propagation, backpropagation, and gradient descent optimization.

- **Forward pass**: Computes the output of the network by passing input data through each layer.
- **Backward pass (Backpropagation)**: Adjusts the weights and biases by calculating the gradient of the loss function.
- **Activation functions**: ReLU and Sigmoid functions are used to introduce non-linearity in the network.
- **Loss function**: Mean squared error (MSE) is used to compute the difference between the predicted output and true labels.