"""Phase 1 verification: run the exact hand-traced network through our NumPy engine."""

import numpy as np
from nn import Linear, MSELoss, Network, Sigmoid


def main():
    # Single data point: 2 inputs, 1 target
    x = np.array([[1.0, 0.5]])
    y = np.array([[1.0]])

    # Build the exact Phase 1 architecture: 2 -> 2 (Sigmoid) -> 1
    l1 = Linear(2, 2)
    l1.W = np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64)
    l1.b = np.array([[0.1, -0.2]], dtype=np.float64)

    sig = Sigmoid()

    l2 = Linear(2, 1)
    l2.W = np.array([[0.5], [-0.4]], dtype=np.float64)
    l2.b = np.array([[0.2]], dtype=np.float64)

    net = Network([l1, sig, l2])
    loss_fn = MSELoss()

    # Forward pass (step-by-step so we can inspect intermediates)
    z1 = l1.forward(x)
    a1 = sig.forward(z1)
    y_hat = l2.forward(a1)
    loss = loss_fn.forward(y_hat, y)

    # Backward pass
    dy_hat = loss_fn.backward()
    da1 = l2.backward(dy_hat)
    dz1 = sig.backward(da1)
    dx = l1.backward(dz1)

    print("=== Phase 1: Forward Pass ===")
    print(f"z1    = {np.round(z1, 6)}")
    print(f"a1    = {np.round(a1, 6)}")
    print(f"y_hat = {np.round(y_hat, 6)}")
    print(f"loss  = {loss:.6f}")

    print("\n=== Phase 1: Backward Pass ===")
    print(f"dy_hat = {np.round(dy_hat, 6)}")
    print(f"dW2    =\n{np.round(l2.dW, 6)}")
    print(f"db2    = {np.round(l2.db, 6)}")
    print(f"da1    = {np.round(da1, 6)}")
    print(f"dz1    = {np.round(dz1, 6)}")
    print(f"dW1    =\n{np.round(l1.dW, 6)}")
    print(f"db1    = {np.round(l1.db, 6)}")
    print(f"dx     = {np.round(dx, 6)}")

    # Verify Network.forward() and Network.backward() produce identical results
    y_hat_net = net.forward(x)
    loss_net = loss_fn.forward(y_hat_net, y)
    dx_net = net.backward(loss_fn.backward())
    assert np.allclose(y_hat_net, y_hat)
    assert np.isclose(loss_net, loss)
    assert np.allclose(dx_net, dx)

    # Assert exact match with Phase 1 hand calculations (within 1e-6 rounding)
    assert np.allclose(z1, [[0.5, -0.45]], atol=1e-6)
    assert np.allclose(a1, [[0.622459, 0.389361]], atol=1e-6)
    assert np.allclose(y_hat, [[0.355485]], atol=1e-6)
    assert np.isclose(loss, 0.415399, atol=1e-6)
    assert np.allclose(dy_hat, [[-1.289029]], atol=1e-6)
    assert np.allclose(l2.dW, [[-0.802368], [-0.501897]], atol=1e-6)
    assert np.allclose(l2.db, [[-1.289029]], atol=1e-6)
    assert np.allclose(da1, [[-0.644515, 0.515612]], atol=1e-6)
    assert np.allclose(dz1, [[-0.151463, 0.122591]], atol=1e-6)
    assert np.allclose(l1.dW, [[-0.151463, 0.122591], [-0.075732, 0.061296]], atol=1e-6)
    assert np.allclose(l1.db, [[-0.151463, 0.122591]], atol=1e-6)
    assert np.allclose(dx, [[-0.067070, -0.048326]], atol=1e-6)

    print("\n[PASS] All Phase 1 code outputs match hand calculations!")


if __name__ == "__main__":
    main()
