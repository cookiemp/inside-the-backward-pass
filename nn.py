"""Minimal neural network library built from scratch with NumPy only."""

import numpy as np


class Layer:
    """Base class for all neural network layers."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params_and_grads(self):
        """Yield (param, grad, name) tuples for trainable parameters."""
        return []


class Linear(Layer):
    """Fully connected linear layer: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # Xavier/Glorot-style scale keeps activations well-behaved
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = rng.normal(0.0, scale, size=(in_features, out_features))
        self.b = np.zeros((1, out_features), dtype=np.float64)

        self.x = None
        self.dW = None
        self.db = None
        self.dx = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        # Gradients for weights, bias, and input using cached self.x
        self.dW = self.x.T @ upstream_grad
        self.db = np.sum(upstream_grad, axis=0, keepdims=True)
        self.dx = upstream_grad @ self.W.T
        return self.dx

    def params_and_grads(self):
        yield self.W, self.dW, "W"
        yield self.b, self.db, "b"


class Sigmoid(Layer):
    """Sigmoid activation: sigma(z) = 1 / (1 + exp(-z))."""

    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        self.out = out
        return self.out

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        return upstream_grad * self.out * (1.0 - self.out)


class ReLU(Layer):
    """Rectified Linear Unit activation: max(0, x)."""

    def __init__(self):
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        return upstream_grad * (self.x > 0).astype(np.float64)


class Tanh(Layer):
    """Hyperbolic tangent activation: tanh(x)."""

    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        return upstream_grad * (1.0 - self.out ** 2)


class MSELoss:
    """Mean Squared Error loss: mean((y_pred - y_true)^2)."""

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        n = self.y_pred.size
        return 2.0 * (self.y_pred - self.y_true) / n


class BCELoss:
    """Binary Cross-Entropy loss for probabilities in (0, 1)."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        self.y_true = y_true
        loss = -(
            y_true * np.log(self.y_pred)
            + (1.0 - y_true) * np.log(1.0 - self.y_pred)
        )
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        n = self.y_pred.size
        return (
            -(self.y_true / self.y_pred)
            + (1.0 - self.y_true) / (1.0 - self.y_pred)
        ) / n


class Network:
    """Sequential container of layers."""

    def __init__(self, layers):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, lr: float) -> None:
        """Update all trainable parameters using vanilla gradient descent."""
        for layer in self.layers:
            for param, grad, _ in layer.params_and_grads():
                param -= lr * grad

    def named_parameters(self):
        """Yield (layer_idx, layer, param, grad, name) for all parameters."""
        for idx, layer in enumerate(self.layers):
            for param, grad, name in layer.params_and_grads():
                yield idx, layer, param, grad, name
