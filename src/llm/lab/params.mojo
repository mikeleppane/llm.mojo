# Parameter zero/update helpers for the lab, one per main-line layer type.
#
# The lab has no optimizer abstraction (AdamW and the real optimizer interface
# come later, designed against the real GPT). Instead the model exposes explicit
# zero_grad() and apply_sgd(lr) that recurse into their children; these helpers
# are the leaves of that recursion, enumerating each layer type's Parameters by
# hand. Verbose on purpose — enumerating every parameter tensor once is the
# chapter's inventory of what a Transformer owns. Nothing here touches the main
# line: the helpers reach into the public .weight/.bias/.table Parameters the
# main-line layers already expose and call the existing free sgd_step.

from llm.nn.embedding import Embedding
from llm.nn.layernorm import LayerNorm
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.transformer.attention import MultiHeadAttention
from llm.training.optimizer import sgd_step


def zero_linear(mut lin: Linear):
    # Zero a Linear's weight and bias grads. Mutates in place; cannot raise.
    lin.weight.zero_grad()
    lin.bias.zero_grad()


def sgd_linear(mut lin: Linear, lr: Float64) raises:
    # One SGD step on a Linear's weight and bias. Mutates the values; raises on a
    # shape mismatch (via sgd_step — never for a well-formed layer).
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
    # The fused-QKV self-attention owns two Linears (qkv, proj).
    zero_linear(a.qkv)
    zero_linear(a.proj)


def sgd_mha(mut a: MultiHeadAttention, lr: Float64) raises:
    sgd_linear(a.qkv, lr)
    sgd_linear(a.proj, lr)


def zero_embedding(mut e: Embedding):
    e.table.zero_grad()


def sgd_embedding(mut e: Embedding, lr: Float64) raises:
    sgd_step(e.table.value, e.table.grad, lr)
