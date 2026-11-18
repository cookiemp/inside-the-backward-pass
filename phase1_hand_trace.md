# Phase 1: Hand-Trace Reference (On Paper)

## 1. Network Architecture & Fixed Parameters

- **Architecture:** `2 inputs (x1, x2)` -> `2 hidden neurons with Sigmoid (h1, h2)` -> `1 output (y_hat)` -> `MSE Loss`
- **Data Point:** `x = [1.0, 0.5]`, `y = 1.0`
- **Hidden Layer (`Linear1`):**
  - `W1 = [[0.2, -0.3], [0.4, 0.1]]`
  - `b1 = [0.1, -0.2]`
- **Output Layer (`Linear2`):**
  - `W2 = [[0.5], [-0.4]]`
  - `b2 = [0.2]`

---

## 2. Forward Pass (Left -> Right)

1. **Hidden raw scores (`z1 = x @ W1 + b1`):**
   - `z1_1 = (1.0)(0.2) + (0.5)(0.4) + 0.1 = 0.2 + 0.2 + 0.1 = 0.5`
   - `z1_2 = (1.0)(-0.3) + (0.5)(0.1) - 0.2 = -0.3 + 0.05 - 0.2 = -0.45`

2. **Hidden activations (`h = sigmoid(z1) = 1 / (1 + e^(-z1))`):**
   - `h1 = 1 / (1 + e^(-0.5)) = 1 / 1.606531 = 0.622459`
   - `h2 = 1 / (1 + e^(0.45)) = 1 / 2.568312 = 0.389361`

3. **Output prediction (`y_hat = h @ W2 + b2`):**
   - `y_hat = (0.622459)(0.5) + (0.389361)(-0.4) + 0.2 = 0.311230 - 0.155744 + 0.2 = 0.355485`

4. **MSE Loss (`L = (y_hat - y)^2`):**
   - `L = (0.355485 - 1.0)^2 = (-0.644515)^2 = 0.415399`

---

## 3. Backward Pass (Right -> Left)

1. **Loss gradient (`dL/dy_hat = 2 * (y_hat - y)`):**
   - `dL/dy_hat = 2 * (0.355485 - 1.0) = -1.289029`

2. **Output Layer (`Linear2`) gradients:**
   - `dL/dW2 = [h1 * dL/dy_hat, h2 * dL/dy_hat]^T`
     - `dL/dw2_1 = 0.622459 * (-1.289029) = -0.802368`
     - `dL/dw2_2 = 0.389361 * (-1.289029) = -0.501897`
   - `dL/db2 = dL/dy_hat = -1.289029`
   - `dL/dh = dL/dy_hat * W2^T`
     - `dL/dh1 = (-1.289029)(0.5) = -0.644515`
     - `dL/dh2 = (-1.289029)(-0.4) = +0.515612`

3. **Sigmoid gradients (`dL/dz1 = dL/dh * h * (1 - h)`):**
   - `dL/dz1_1 = (-0.644515) * 0.622459 * (1 - 0.622459) = -0.151463`
   - `dL/dz1_2 = (+0.515612) * 0.389361 * (1 - 0.389361) = +0.122591`

4. **Hidden Layer (`Linear1`) gradients:**
   - `dL/dW1 = x^T @ dL/dz1`:
     - `[[1.0 * (-0.151463), 1.0 * 0.122591], [0.5 * (-0.151463), 0.5 * 0.122591]]`
     - `= [[-0.151463, 0.122591], [-0.075732, 0.061296]]`
   - `dL/db1 = dL/dz1 = [-0.151463, 0.122591]`
   - `dL/dx = dL/dz1 @ W1^T`:
     - `dL/dx1 = (-0.151463)(0.2) + (0.122591)(-0.3) = -0.067070`
     - `dL/dx2 = (-0.151463)(0.4) + (0.122591)(0.1) = -0.048326`
