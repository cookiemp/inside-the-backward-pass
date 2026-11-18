# Inside the Backward Pass — Neural Network from Scratch

A minimal neural network library built from scratch using **NumPy and Matplotlib only** (no PyTorch, TensorFlow, or autograd).

## Project Files

- `nn.py` — Core engine (`Linear`, `Sigmoid`, `ReLU`, `Tanh`, `MSELoss`, `BCELoss`, `Network`).
- `hand_trace/` — Handwritten Phase 1 paper calculations (`page1_forward.jpg`, `page2_backward_part1.jpg`, `page3_backward_part2.jpg`).
- `phase1_hand_trace.md` — Typed reference of the Phase 1 calculations (`2 -> 2 (Sigmoid) -> 1`, `MSELoss`).
- `phase1_verify.py` — Runs the exact Phase 1 network and asserts every forward value and gradient matches the hand trace.
- `grad_check.py` — Numerical gradient checking via central finite differences (`rel_error < 1e-6`).
- `train_nonlinear.py` — Trains a linear baseline vs. an MLP on a pure-NumPy 2D Two Moons dataset (80/20 train/test split) and saves plots.
- `REPORT.md` — Full technical report summarizing all phases and results.

## How to Run

```bash
pip install numpy matplotlib
python phase1_verify.py
python grad_check.py
python train_nonlinear.py
```

## Results

### Phase 1 & Phase 3 Verification
- `hand_trace/`: Handwritten paper calculations for the `2 -> 2 (Sigmoid) -> 1` network (`MSELoss`, `x = [1.0, 0.5]`, `y = 1.0`).
- `phase1_verify.py`: All forward values (`z1`, `a1`, `y_hat`, `loss`) and backward gradients (`dW2`, `db2`, `da1`, `dz1`, `dW1`, `db1`, `dx`) match the handwritten trace (`[PASS]`).
- `grad_check.py`: All parameters across `Linear`, `Sigmoid`, `ReLU`, `Tanh`, `MSELoss`, and `BCELoss` pass with relative error between `1.438e-11` and `1.378e-09` (well below `1e-6`).

| Page 1: Forward Pass | Page 2: Loss &amp; Output Gradients | Page 3: Hidden Gradients |
| :---: | :---: | :---: |
| ![Page 1](hand_trace/page1_forward.jpg) | ![Page 2](hand_trace/page2_backward_part1.jpg) | ![Page 3](hand_trace/page3_backward_part2.jpg) |

### Non-Linear Classification (Two Moons, 240 Train / 60 Test)

| Model | Final Loss | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| Linear Baseline (`2 -> 1`) | `0.2681` | `87.5%` | `88.3%` |
| MLP with Hidden Layers (`2 -> 16 -> 16 -> 1`) | `0.0022` | `100.0%` | `100.0%` |

| Decision Boundaries | Training Loss |
| :---: | :---: |
| ![Decision Boundaries](decision_boundary.png) | ![Training Loss](training_loss.png) |
