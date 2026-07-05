"""Reference AdamW + LR-schedule values for the Mojo training tests.

Provenance, not a test-time dependency: run once by hand, its printed numbers
frozen as literals into tests/test_adamw.mojo and tests/test_schedule.mojo.
Independent NumPy math; nothing under src/ or the suite imports this.

Run:  pixi run python tests/oracles/adamw_reference.py
"""

import numpy as np


def adamw_step(value, grad, m, v, t, lr, beta1, beta2, eps, weight_decay):
    """One in-place-style AdamW step (decoupled decay). Returns new tensors.

        m <- b1*m + (1-b1)*g
        v <- b2*v + (1-b2)*g^2
        mhat = m/(1-b1^t)   vhat = v/(1-b2^t)        (t starts at 1)
        value <- value - lr*( mhat/(sqrt(vhat)+eps) + weight_decay*value )
    """
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad * grad)
    mhat = m / (1.0 - beta1**t)
    vhat = v / (1.0 - beta2**t)
    value = value - lr * (mhat / (np.sqrt(vhat) + eps) + weight_decay * value)
    return value, m, v


def hexf(x):
    return np.float64(x).tobytes()[::-1].hex()  # not used; kept for reference


def run_scalar_case(value0, grads, lr, beta1, beta2, eps, wd, label):
    print(f"\n# --- {label}: lr={lr} b1={beta1} b2={beta2} eps={eps} wd={wd}")
    value = np.array(value0, dtype=np.float64)
    m = np.zeros_like(value)
    v = np.zeros_like(value)
    for i, g in enumerate(grads, start=1):
        grad = np.array(g, dtype=np.float64)
        value, m, v = adamw_step(value, grad, m, v, i, lr, beta1, beta2, eps, wd)
        print(f"step {i}: value={value.tolist()}")
        print(f"         m={m.tolist()}")
        print(f"         v={v.tolist()}")


def main():
    np.set_printoptions(precision=17)

    # Case A: 2x2 tensor, decay ON, constant gradient, several steps.
    value0 = [[0.5, -1.0], [2.0, 0.25]]
    grad = [[0.1, -0.2], [0.3, 0.05]]
    run_scalar_case(
        value0, [grad, grad, grad], lr=0.01, beta1=0.9, beta2=0.95,
        eps=1e-8, wd=0.1, label="A decay=0.1 constant grad 3 steps",
    )

    # Case B: same, decay OFF.
    run_scalar_case(
        value0, [grad, grad, grad], lr=0.01, beta1=0.9, beta2=0.95,
        eps=1e-8, wd=0.0, label="B decay=0.0 constant grad 3 steps",
    )

    # Case C: varying gradient, decay ON, 4 steps (catches bias-correction).
    grads = [
        [[0.1, -0.2], [0.3, 0.05]],
        [[-0.4, 0.1], [0.0, 0.2]],
        [[0.2, 0.2], [-0.1, -0.3]],
        [[0.05, -0.05], [0.15, 0.1]],
    ]
    run_scalar_case(
        value0, grads, lr=0.05, beta1=0.9, beta2=0.95,
        eps=1e-8, wd=0.1, label="C decay=0.1 varying grad 4 steps",
    )

    # Step-1 hand check: with m=v=0, after one step
    #   m1 = (1-b1) g, v1 = (1-b2) g^2
    #   mhat = m1/(1-b1) = g ; vhat = v1/(1-b2) = g^2
    #   value <- value - lr*( g/(|g|+eps) + wd*value )
    # so the bias correction at t=1 exactly cancels the moment scaling: the
    # adaptive term is g/(|g|+eps) ~ sign(g). Verify:
    g = 0.3
    lr, wd, eps = 0.01, 0.1, 1e-8
    v0 = 2.0
    m1 = (1 - 0.9) * g
    v1 = (1 - 0.95) * g * g
    mhat = m1 / (1 - 0.9)
    vhat = v1 / (1 - 0.95)
    step1 = v0 - lr * (mhat / (np.sqrt(vhat) + eps) + wd * v0)
    print("\n# --- step-1 hand check (g=0.3, v0=2.0, lr=0.01, wd=0.1)")
    print(f"mhat={mhat} (==g), vhat={vhat} (==g^2), value1={step1!r}")
    # decay-off comparison for the same
    step1_nodecay = v0 - lr * (mhat / (np.sqrt(vhat) + eps))
    print(f"value1 (wd=0) ={step1_nodecay!r}")

    # Decoupled-decay pin: g=0 => moments stay 0, value shrinks by lr*wd*value.
    print("\n# --- decoupled decay g=0")
    val = 3.0
    for i in range(1, 4):
        # m,v stay 0; mhat/(sqrt(vhat)+eps) = 0/(0+eps) = 0
        val = val - lr * (0.0 + wd * val)
        print(f"step {i}: value={val!r}  (== prev*(1-lr*wd))")
    print(f"expected factor per step (1-lr*wd) = {1 - lr*wd!r}")

    # Schedule reference: lr_at(step, peak, warmup, max_steps, min_lr)
    print("\n# === schedule ===")
    peak, warmup, max_steps, min_lr = 1.0, 10, 100, 0.1
    import math

    def lr_at(step, peak, warmup, max_steps, min_lr):
        if step < warmup:
            return peak * step / warmup  # linear warmup from 0 (step 0 -> 0)
        if step >= max_steps:
            return min_lr
        # cosine decay from peak to min_lr over [warmup, max_steps]
        progress = (step - warmup) / (max_steps - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak - min_lr) * cos

    for s in [0, 1, 5, 10, 55, 100, 150]:
        print(f"lr_at({s}) = {lr_at(s, peak, warmup, max_steps, min_lr)!r}")
    # warmup_steps = 0 degenerate: step 0 should be peak (no warmup phase)
    print("warmup=0:")
    for s in [0, 50, 100]:
        print(f"lr_at({s}) = {lr_at(s, peak, 0, max_steps, min_lr)!r}")


if __name__ == "__main__":
    main()
