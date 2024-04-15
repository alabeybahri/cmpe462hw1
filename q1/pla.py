import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, X, y, weights=None):
        self.X = X
        self.y = y
        self.weights = weights if weights is not None else np.zeros(self.X.shape[1])
        self.iterations = 0

    def fit(self):
        self.iterations = 0
        while True:
            all_correctly_classified = True
            indices = np.random.permutation(self.X.shape[0])
            for i in indices:
                if np.sign(np.dot(self.weights, self.X[i])) != self.y[i]:
                    all_correctly_classified = False
                    self.weights += self.y[i] * self.X[i]
                    break
            self.iterations += 1
            if all_correctly_classified:
                break

    def predict(self):
        return np.sign(np.dot(self.X, self.weights))

    def plot_decision_boundary(self, title="Decision Boundary"):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.X[:, 1],
            self.X[:, 2],
            c=self.y,
            cmap="bwr",
            marker="o",
            label="Classes",
        )
        x1 = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 100)
        x2 = -(self.weights[1] * x1 + self.weights[0]) / self.weights[2]
        plt.plot(x1, x2, "k-", label="Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    data_large = np.load("PLA_data/data_large.npy")
    label_large = np.load("PLA_data/label_large.npy")
    data_small = np.load("PLA_data/data_small.npy")
    label_small = np.load("PLA_data/label_small.npy")

    perceptron_large = Perceptron(data_large, label_large)
    perceptron_large.fit()
    predictions_large = perceptron_large.predict()
    perceptron_large.plot_decision_boundary(title="Large Dataset")
    perceptron_small = Perceptron(data_small, label_small)
    perceptron_small.fit()
    predictions_small = perceptron_small.predict()
    perceptron_small.plot_decision_boundary(title="Small Dataset")
    # print(perceptron_large.weights)
    # print(perceptron_small.weights)
    # print(perceptron_large.iterations)
    # print(perceptron_small.iterations)
