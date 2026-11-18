"""Train a scratch-built MLP on a non-linear 2D Two Moons dataset and plot results."""

import numpy as np
from nn import BCELoss, Linear, Network, Sigmoid, Tanh

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def make_two_moons(n_samples: int = 300, noise: float = 0.12, seed: int = 42):
    """Generate a 2D Two Moons dataset in pure NumPy."""
    rng = np.random.default_rng(seed)
    n_per_moon = n_samples // 2

    theta1 = np.linspace(0, np.pi, n_per_moon)
    moon1 = np.column_stack([np.cos(theta1), np.sin(theta1)])

    theta2 = np.linspace(0, np.pi, n_samples - n_per_moon)
    moon2 = np.column_stack([1.0 - np.cos(theta2), 0.5 - np.sin(theta2)])

    X = np.vstack([moon1, moon2]) + rng.normal(0.0, noise, size=(n_samples, 2))
    y = np.vstack([np.zeros((n_per_moon, 1)), np.ones((n_samples - n_per_moon, 1))])

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def train_model(net: Network, X: np.ndarray, y: np.ndarray, epochs: int, lr: float):
    """Train a network with full-batch gradient descent and return loss history."""
    loss_fn = BCELoss()
    losses = []

    for epoch in range(epochs):
        y_pred = net.forward(X)
        loss = loss_fn.forward(y_pred, y)
        losses.append(loss)

        net.backward(loss_fn.backward())
        net.step(lr)

        if (epoch + 1) % 500 == 0 or epoch == 0:
            acc = np.mean((y_pred >= 0.5) == y) * 100.0
            print(f"  Epoch {epoch + 1:4d}/{epochs} | Loss: {loss:.4f} | Acc: {acc:5.1f}%")

    final_acc = float(np.mean((net.forward(X) >= 0.5) == y) * 100.0)
    return losses, final_acc


def plot_results(X, y, linear_net, mlp_net, linear_losses, mlp_losses):
    """Save decision boundary comparison and training loss curves."""
    if not HAS_MATPLOTLIB:
        print("\n[WARN] matplotlib is not installed; skipping plots.")
        return

    # 1. Decision boundary comparison
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Linear Model (No Hidden Layer)", "MLP (Hidden Layers with Tanh)"]

    for ax, model, title in zip(axes, [linear_net, mlp_net], titles):
        probs = model.forward(grid).reshape(xx.shape)
        ax.contourf(xx, yy, probs, levels=25, cmap="RdBu", alpha=0.75)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
        ax.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c="#b2182b", edgecolors="k", s=28, label="Class 0")
        ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c="#2166ac", edgecolors="k", s=28, label="Class 1")
        acc = np.mean((model.forward(X) >= 0.5) == y) * 100.0
        ax.set_title(f"{title}\nAccuracy: {acc:.1f}%")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150)
    plt.close(fig)

    # 2. Loss curve comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(linear_losses, label="Linear Model (No Hidden Layer)", linestyle="--")
    ax.plot(mlp_losses, label="MLP (With Hidden Layers)", linewidth=2)
    ax.set_title("Training Loss on Two Moons (BCELoss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.close(fig)

    print("\nSaved plots: decision_boundary.png, training_loss.png")


def main():
    X, y = make_two_moons(n_samples=300, noise=0.12, seed=42)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    rng = np.random.default_rng(7)

    print("=== Training Linear Baseline (2 -> 1) ===")
    linear_net = Network([Linear(2, 1, rng=rng), Sigmoid()])
    linear_losses, linear_train_acc = train_model(linear_net, X_train, y_train, epochs=2000, lr=0.5)
    linear_test_acc = float(np.mean((linear_net.forward(X_test) >= 0.5) == y_test) * 100.0)

    print("\n=== Training MLP with Hidden Layers (2 -> 16 -> 16 -> 1) ===")
    mlp_net = Network([
        Linear(2, 16, rng=rng),
        Tanh(),
        Linear(16, 16, rng=rng),
        Tanh(),
        Linear(16, 1, rng=rng),
        Sigmoid(),
    ])
    mlp_losses, mlp_train_acc = train_model(mlp_net, X_train, y_train, epochs=2000, lr=0.5)
    mlp_test_acc = float(np.mean((mlp_net.forward(X_test) >= 0.5) == y_test) * 100.0)

    print(
        f"\nSummary:\n"
        f"  Linear Baseline -> Train Acc: {linear_train_acc:.1f}% | Test Acc: {linear_test_acc:.1f}%\n"
        f"  MLP (Hidden)    -> Train Acc: {mlp_train_acc:.1f}% | Test Acc: {mlp_test_acc:.1f}%"
    )
    plot_results(X, y, linear_net, mlp_net, linear_losses, mlp_losses)


if __name__ == "__main__":
    main()
