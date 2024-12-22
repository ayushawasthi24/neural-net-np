import argparse
import numpy as np
from nn import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(
        description="Train and test a simple neural network."
    )

    # Add arguments
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="List of layers in the neural network (e.g., 2 5 1 for 2 input, 5 hidden, 1 output).",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "sigmoid"],
        default="relu",
        help="Activation function to use (default: relu).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs for training (default: 1000).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for training (default: 0.01).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples in the dataset (default: 100).",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=2,
        help="Number of features in the dataset (default: 2).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Flag to visualize predictions and training loss.",
    )

    args = parser.parse_args()

    # Generate dataset
    X = np.random.rand(args.samples, args.features)
    y = np.array([[np.sum(x)] for x in X])

    # Initialize and train neural network
    nn = NeuralNetwork(args.layers, activation=args.activation, visualize=args.visualize)
    print("Starting training...")
    nn.train(X, y, epochs=args.epochs, learning_rate=args.learning_rate)

    # Visualize predictions if the flag is set
    if args.visualize:
        nn.visualize_predictions(X, y)

    # Predict on the first 5 samples
    predictions = nn.predict(X[:5])
    print("Inputs:", X[:5])
    print("Predictions:", predictions)
    print("True Values:", y[:5])
    print("Loss:", np.mean(np.square(predictions - y[:5])))


if __name__ == "__main__":
    main()
