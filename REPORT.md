# Capstone Report: Inside the Backward Pass

## 1. Goal & Architecture
We built a minimal neural network library from scratch using **NumPy and Matplotlib only** (`nn.py`), proved its gradients are mathematically exact, and used it to solve a non-linear 2D classification problem (**Two Moons**) where a linear model fails.

### Implemented Modules (`nn.py`)
Each layer implements `forward(x)` (computing outputs and caching any values needed for derivatives) and `backward(upstream_grad)` (computing parameter gradients and returning the gradient with respect to the layer's input):
- **`Linear`**: Computes `y = x @ W + b`, caches `x`, and in `backward(upstream_grad)` computes `dW = x.T @ upstream_grad`, `db = sum(upstream_grad, axis=0)`, and returns `dx = upstream_grad @ W.T`.
- **`Sigmoid`**: Computes $\sigma(z) = \frac{1}{1 + e^{-z}}$, caches output $a$, and returns `upstream_grad * a * (1 - a)`.
- **`ReLU`**: Computes $\max(0, x)$, caches $x$, and returns `upstream_grad * (x > 0)`.
- **`Tanh`**: Computes $\tanh(x)$, caches output $a$, and returns `upstream_grad * (1 - a^2)`.
- **`MSELoss`**: Computes $\text{mean}((\hat{y} - y)^2)$ and returns $\frac{2(\hat{y} - y)}{N}$.
- **`BCELoss`**: Computes binary cross-entropy with numerical clipping (`eps=1e-12`) and returns its derivative with respect to $\hat{y}$.
- **`Network`**: Calls `forward` on layers in order and `backward` in reverse order, with `step(lr)` for gradient descent.

---

## 2. How the Backward Pass Works
During training, the **forward pass** computes predictions from left to right and caches each layer's inputs/activations. The **backward pass** applies the **chain rule** from right to left:
1. The loss function produces the initial sensitivity $\frac{\partial L}{\partial \hat{y}}$—how much the loss changes when the prediction changes.
2. Each layer receives `upstream_grad` ($\frac{\partial L}{\partial \text{output}}$) from the layer ahead of it.
3. Using its cached forward values, the layer computes:
   - **Parameter gradients** ($\frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}$), which tell gradient descent which direction to nudge each weight to lower the loss.
   - **Input gradient** ($\frac{\partial L}{\partial \text{input}}$), which it passes backward as the `upstream_grad` for the previous layer.

---

## 3. Phase 1: Hand Trace vs. Code Verification
Using the fixed `2 -> 2 (Sigmoid) -> 1` network with `MSELoss`, `x = [1.0, 0.5]`, and `y = 1.0` (documented in [hand_trace/](hand_trace/)), running `python phase1_verify.py` matched every hand-calculated value:

| Quantity | Hand Calculation | Code Output (`phase1_verify.py`) |
| :--- | :--- | :--- |
| `z1` | `[0.5, -0.45]` | `[[0.5, -0.45]]` |
| `a1` (`h1, h2`) | `[0.622459, 0.389361]` | `[[0.622459, 0.389361]]` |
| `y_hat` | `0.355485` | `[[0.355485]]` |
| `loss` | `0.415399` | `0.415399` |
| `dL/dy_hat` | `-1.289029` | `[[-1.289029]]` |
| `dL/dW2` | `[-0.802368, -0.501897]^T` | `[[-0.802368], [-0.501897]]` |
| `dL/db2` | `-1.289029` | `[[-1.289029]]` |
| `dL/da1` | `[-0.644515, 0.515612]` | `[[-0.644515, 0.515612]]` |
| `dL/dz1` | `[-0.151463, 0.122591]` | `[[-0.151463, 0.122591]]` |
| `dL/dW1` | `[[-0.151463, 0.122591], [-0.075732, 0.061296]]` | `[[-0.151463, 0.122591], [-0.075732, 0.061296]]` |
| `dL/db1` | `[-0.151463, 0.122591]` | `[[-0.151463, 0.122591]]` |

**Discrepancies:** None (`0.0` difference within 6-decimal rounding; all assertions pass).

---

## 4. Phase 3: Numerical Gradient Check
`grad_check.py` compares analytical backprop gradients against central finite differences:
$$\frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}, \qquad \epsilon = 10^{-6}$$

All parameters across both the Phase 1 network (`Linear -> Sigmoid -> Linear + MSELoss`) and a deeper network (`Linear -> ReLU -> Linear -> Tanh -> Linear -> Sigmoid + BCELoss`) passed well below the `1e-6` threshold:
- Phase 1 network relative errors: `1.438e-11` to `1.258e-10` (**PASS**)
- Deep MLP (`ReLU`, `Tanh`, `Sigmoid`, `BCELoss`) relative errors: `1.402e-10` to `1.378e-09` (**PASS**)

---

## 5. Non-Linear Dataset Results (Two Moons)
Running `python train_nonlinear.py` on a 300-point 2D Two Moons dataset (80/20 split: 240 train / 60 test, 2000 epochs, `lr = 0.5`, `BCELoss`):
- **Linear Baseline (`2 -> 1`):** Final Loss = `0.2681`, Train Accuracy = **87.5%**, Test Accuracy = **88.3%** (a single straight line cannot separate the interlocking moons).
- **MLP with Hidden Layers (`2 -> 16 -> 16 -> 1`, `Tanh` + `Sigmoid`):** Final Loss = `0.0022`, Train Accuracy = **100.0%**, Test Accuracy = **100.0%** (the hidden layers learn a smooth, curved decision boundary).

Generated plots:
- `decision_boundary.png` — Side-by-side comparison of the linear vs. MLP decision boundaries.
- `training_loss.png` — Loss curves over 2000 epochs.
