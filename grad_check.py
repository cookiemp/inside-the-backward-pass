"""Phase 3: Numerical gradient checking to prove backprop gradients are exact."""

import numpy as np
from nn import BCELoss, Linear, MSELoss, Network, ReLU, Sigmoid, Tanh


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 error between two gradient arrays."""
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + np.linalg.norm(b)
    if den < 1e-12:
        return 0.0
    return float(num / den)


def gradient_check(
    network: Network,
    loss_fn,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
    tol: float = 1e-6,
    verbose: bool = True,
):
    """Compare analytical backprop gradients against central-difference numerical gradients."""
    y_pred = network.forward(x)
    _ = loss_fn.forward(y_pred, y)
    network.backward(loss_fn.backward())

    # Snapshot analytical gradients before perturbing parameters
    snapshots = []
    for layer_idx, _, param, grad, name in network.named_parameters():
        snapshots.append((layer_idx, param, grad.copy(), name))

    results = []
    all_passed = True

    for layer_idx, param, bp_grad, name in snapshots:
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]

            param[idx] = orig + eps
            loss_plus = loss_fn.forward(network.forward(x), y)

            param[idx] = orig - eps
            loss_minus = loss_fn.forward(network.forward(x), y)

            param[idx] = orig
            num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            it.iternext()

        err = relative_error(num_grad, bp_grad)
        passed = err < tol
        all_passed = all_passed and passed
        results.append((f"layer_{layer_idx}.{name}", err, passed))

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  layer_{layer_idx}.{name:<3} | rel_error = {err:.3e} | [{status}]")

    return all_passed, results


def main():
    rng = np.random.default_rng(42)

    print("=== Gradient Check 1: Phase 1 Network (Linear -> Sigmoid -> Linear + MSELoss) ===")
    l1 = Linear(2, 2)
    l1.W = np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64)
    l1.b = np.array([[0.1, -0.2]], dtype=np.float64)
    l2 = Linear(2, 1)
    l2.W = np.array([[0.5], [-0.4]], dtype=np.float64)
    l2.b = np.array([[0.2]], dtype=np.float64)
    net1 = Network([l1, Sigmoid(), l2])
    x1 = np.array([[1.0, 0.5]])
    y1 = np.array([[1.0]])
    ok1, _ = gradient_check(net1, MSELoss(), x1, y1)

    print("\n=== Gradient Check 2: Deep MLP (ReLU -> Tanh -> Sigmoid + BCELoss) ===")
    x2 = rng.normal(size=(5, 3))
    y2 = np.array([[1.0], [0.0], [1.0], [0.0], [1.0]])
    net2 = Network([
        Linear(3, 4, rng=rng),
        ReLU(),
        Linear(4, 3, rng=rng),
        Tanh(),
        Linear(3, 1, rng=rng),
        Sigmoid(),
    ])
    ok2, _ = gradient_check(net2, BCELoss(), x2, y2)

    assert ok1 and ok2, "One or more gradient checks failed!"
    print("\n[PASS] All layers and loss functions passed gradient check (< 1e-6)!")


if __name__ == "__main__":
    main()
