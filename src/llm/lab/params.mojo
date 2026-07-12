"""Parameter zero/update helpers, one per layer type.

The quarantined encoder-decoder lab has no optimizer abstraction: models expose
explicit zero_grad() and apply_sgd(lr) that recurse into their children, and
these helpers are the leaves, enumerating each layer type's parameters by hand.
"""

from llm.nn.embedding import Embedding
from llm.nn.layernorm import LayerNorm
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.transformer.attention import MultiHeadAttention
from llm.training.optimizer import sgd_step


def zero_linear(mut lin: Linear):
    """Zero a Linear's weight and bias grads.

    Args:
        lin: Layer whose gradients are zeroed in place.
    """
    lin.weight.zero_grad()
    lin.bias.zero_grad()


def sgd_linear(mut lin: Linear, lr: Float64) raises:
    """Apply one SGD step to a Linear's weight and bias.

    Args:
        lin: Layer whose values are updated in place.
        lr: Learning rate.

    Raises:
        Error: On a shape mismatch (never for a well-formed layer).
    """
    sgd_step(lin.weight.value, lin.weight.grad, lr)
    sgd_step(lin.bias.value, lin.bias.grad, lr)


def zero_layernorm(mut ln: LayerNorm):
    ln.weight.zero_grad()
    ln.bias.zero_grad()


def sgd_layernorm(mut ln: LayerNorm, lr: Float64) raises:
    sgd_step(ln.weight.value, ln.weight.grad, lr)
    sgd_step(ln.bias.value, ln.bias.grad, lr)


def zero_mlp(mut m: MLP):
    zero_linear(m.up)
    zero_linear(m.down)


def sgd_mlp(mut m: MLP, lr: Float64) raises:
    sgd_linear(m.up, lr)
    sgd_linear(m.down, lr)


def zero_mha(mut a: MultiHeadAttention):
    """Zero the fused-QKV self-attention's two Linears (qkv, proj)."""
    zero_linear(a.qkv)
    zero_linear(a.proj)


def sgd_mha(mut a: MultiHeadAttention, lr: Float64) raises:
    sgd_linear(a.qkv, lr)
    sgd_linear(a.proj, lr)


def zero_embedding(mut e: Embedding):
    e.table.zero_grad()


def sgd_embedding(mut e: Embedding, lr: Float64) raises:
    sgd_step(e.table.value, e.table.grad, lr)
