import matplotlib.pyplot as plt
import time

class NeuralNetworkVisualizer:
    def __init__(self):
        """Initialize the visualizer with a figure for dynamic plotting."""
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_title("Training Loss Over Epochs")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        (self.line,) = self.ax.plot([], [], label="Loss", color="blue")
        self.ax.legend()
        self.losses = []
        self.epochs = []
        plt.ion()  # Turn on interactive mode
        plt.show()

    def update_loss_plot(self, epoch, loss):
        """Update the loss plot dynamically during training."""
        self.losses.append(loss)
        self.epochs.append(epoch)
        self.line.set_data(self.epochs, self.losses)
        self.ax.set_xlim(0, max(self.epochs) + 1)
        self.ax.set_ylim(0, max(self.losses) * 1.1)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        time.sleep(0.01)  # Allow the plot to refresh

    def finalize_plot(self):
        """Finalize and keep the plot open after training."""
        plt.ioff()
        plt.show()

    @staticmethod
    def plot_predictions(y_true, y_pred):
        """Visualize predictions versus true values."""
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(y_true)), y_true, label="True Values", color="blue")
        plt.scatter(range(len(y_true)), y_pred, label="Predictions", color="red")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title("Predictions vs True Values")
        plt.legend()
        plt.grid(True)
        plt.show()
