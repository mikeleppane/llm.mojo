"""Reference values for the attention-core Mojo tests.

Provenance, not a test-time dependency: run **once** by hand, its printed numbers
frozen as literals into ``tests/test_attention_core.mojo`` with a comment
pointing back here (same arrangement as ``nn_reference.py`` and
``gpt2_reference_encoder.py``). The Mojo tests then stay fully offline — nothing
under ``src/`` or the suite imports this file.

Everything is float64 NumPy so the goldens are an independent oracle: the
reference math lives here, the implementation lives in
``src/llm/transformer/attention.mojo``, and the two meet only through the frozen
literals.

The scaled-dot-product core, pinned order:

    scores  = q @ k.T / sqrt(D)      # D = q.shape[1] = d_head
    scores += mask                    # additive, AFTER the scale
    weights = softmax(scores, axis=1) # row-wise, stable
    output  = weights @ v

Run:  pixi run python tests/oracles/attention_reference.py
"""

from __future__ import annotations

import numpy as np


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax (subtract the row max), matching softmax_rows."""
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    return e / e.sum(axis=1, keepdims=True)


def sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray):
    """Return (weights, output) for scaled-dot-product attention, pinned order."""
    d = q.shape[1]
    scores = q @ k.T / np.sqrt(float(d))  # scale first
    scores = scores + mask  # then add the additive mask
    weights = softmax_rows(scores)  # then softmax rows
    output = weights @ v  # then weight the values
    return weights, output


def dump(label: str, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    print(f"{label}:  # shape {arr.shape}")
    for val in flat:
        print(f"    {val!r}")


def main() -> None:
    np.set_printoptions(precision=17)

    # --- Case A: cross-shaped core, T_q=3 != T_k=4, D=2, D_v=2, no mask ---
    # T_q != T_k proves the self/cross split (separate q vs k/v lengths) works,
    # and pins the output shape as [T_q, D_v] = [3, 2] (not [T_k, ...]).
    q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [3, 2]
    k = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])  # [4, 2]
    v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])  # [4, 2]
    no_mask = np.zeros((3, 4))
    w, o = sdpa(q, k, v, no_mask)
    print("# Case A: cross-shaped, no mask")
    dump("A_weights", w)  # [3, 4]
    dump("A_output", o)  # [3, 2]
    print()

    # --- Case B: scale is 1/sqrt(D), not squared, not 1/sqrt(d_model) ---
    # q has a single 1 in column 0, so dot(q, k_row) = k_row[0] regardless of D.
    # Padding q and the keys with extra columns leaves the two dot products
    # fixed at 8 and 2 while D changes 2 -> 4, so the ONLY thing moving the
    # weights is the 1/sqrt(D) factor. weights_D2 = softmax([8, 2]/sqrt(2)),
    # weights_D4 = softmax([8, 2]/sqrt(4)). A squared or d_model scale, or none,
    # gives different numbers.
    q2 = np.array([[1.0, 0.0]])  # [1, 2]
    k2 = np.array([[8.0, 7.0], [2.0, 3.0]])  # [2, 2]; dots = 8, 2
    v2 = np.array([[1.0], [0.0]])  # [2, 1] — output tracks weight[0,0]
    w2, o2 = sdpa(q2, k2, v2, np.zeros((1, 2)))
    q4 = np.array([[1.0, 0.0, 0.0, 0.0]])  # [1, 4]
    k4 = np.array([[8.0, 7.0, 5.0, 5.0], [2.0, 3.0, 9.0, 9.0]])  # dots still 8, 2
    w4, o4 = sdpa(q4, k4, v2, np.zeros((1, 2)))
    print("# Case B: scale test, dots fixed at (8, 2), D = 2 then 4")
    dump("B_weights_D2", w2)  # softmax([8,2]/sqrt(2))
    dump("B_weights_D4", w4)  # softmax([8,2]/sqrt(4))
    print()

    # --- Case C: scale-BEFORE-mask order, small finite mask ---
    # Reuses Case A's q/k/v but adds a mask of small finite entries. The pinned
    # order scales the raw scores THEN adds the mask; a wrong order (add mask,
    # then scale) would divide these mask entries by sqrt(2) too and produce
    # different weights. Small finite entries (not -1e9) make the two orders
    # diverge observably — MASKED_SCORE is too large to tell them apart.
    mask_c = np.array(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0], [-3.0, 0.0, 0.0, 0.0]]
    )
    wc, oc = sdpa(q, k, v, mask_c)
    print("# Case C: scale-before-mask order, small finite mask")
    dump("C_weights", wc)  # [3, 4]
    dump("C_output", oc)  # [3, 2]
    print()

    # --- Case D: hand-worked 2x2, D=1 (scale = 1/sqrt(1) = 1) ---
    # Small enough to work by hand in the test comment; these frozen digits just
    # confirm the hand arithmetic. scores = q @ k.T = [[1, 2], [0, 0]];
    # row 0 weights = softmax([1, 2]) = [1/(1+e), e/(1+e)]; row 1 = [0.5, 0.5].
    qd = np.array([[1.0], [0.0]])  # [2, 1]
    kd = np.array([[1.0], [2.0]])  # [2, 1]
    vd = np.array([[3.0], [5.0]])  # [2, 1]
    wd, od = sdpa(qd, kd, vd, np.zeros((2, 2)))
    print("# Case D: hand-worked 2x2, D=1")
    dump("D_weights", wd)  # [2, 2]
    dump("D_output", od)  # [2, 1]


if __name__ == "__main__":
    main()
