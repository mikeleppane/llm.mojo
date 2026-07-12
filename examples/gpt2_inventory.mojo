"""Build the real GPT-2 124M preset, count its actual floats, and reconcile.

Allocates the preset for real and walks every Parameter, summing value.size(),
then asserts the walked total equals both GPTConfig.parameter_count() and the
literal 124,439,808.

Run manually: it allocates ~1 GB of Float64 weights plus the same again in
gradient buffers (~2 GB resident) and draws ~124M normal samples, so it is not in
the test suite (where the walk-equals-formula invariant is pinned on tiny configs
without the allocation).

    pixi run mojo run -I src examples/gpt2_inventory.mojo
"""

from llm.config import GPTConfig
from llm.transformer.gpt import GPT
from llm.utils.random import Rng


def block_parameter_count(gpt: GPT, i: Int) -> Int:
    """Sum the 12 Parameters of transformer block i.

    Covers two LayerNorms (weight+bias), the fused qkv and proj Linears
    (weight+bias), and the MLP up/down Linears (weight+bias).

    Args:
        gpt: The model.
        i: Block index.

    Returns:
        The block's parameter count.
    """
    var total = 0
    total += gpt.blocks[i].ln1.weight.value.size()
    total += gpt.blocks[i].ln1.bias.value.size()
    total += gpt.blocks[i].attn.qkv.weight.value.size()
    total += gpt.blocks[i].attn.qkv.bias.value.size()
    total += gpt.blocks[i].attn.proj.weight.value.size()
    total += gpt.blocks[i].attn.proj.bias.value.size()
    total += gpt.blocks[i].ln2.weight.value.size()
    total += gpt.blocks[i].ln2.bias.value.size()
    total += gpt.blocks[i].mlp.up.weight.value.size()
    total += gpt.blocks[i].mlp.up.bias.value.size()
    total += gpt.blocks[i].mlp.down.weight.value.size()
    total += gpt.blocks[i].mlp.down.bias.value.size()
    return total


def main() raises:
    """Allocate the 124M preset and assert the walked count matches the formula.
    """
    var cfg = GPTConfig.gpt2_124m()
    print(cfg)
    print("Allocating the 124M preset (~2 GB resident)...")

    var rng = Rng(0)
    var gpt = GPT.init_random(cfg, rng)

    # Component-by-component inventory, counted off the real tensors.
    var wte = gpt.wte.table.value.size()  # V * C
    var wpe = gpt.wpe.table.value.size()  # context_length * C
    var per_block = block_parameter_count(gpt, 0)
    var blocks_total = 0
    for i in range(len(gpt.blocks)):
        blocks_total += block_parameter_count(gpt, i)
    var ln_f = gpt.ln_f.weight.value.size() + gpt.ln_f.bias.value.size()

    print("")
    print("Component                         Parameters")
    print("------------------------------------------------")
    print("token embedding    (wte, V*C)    ", wte)
    print("positional embed   (wpe, T*C)    ", wpe)
    print("per block          (x", len(gpt.blocks), ")         ", per_block)
    print("all blocks                       ", blocks_total)
    print("final LayerNorm    (ln_f, 2C)    ", ln_f)
    print("LM head            (tied)                 0")
    print("------------------------------------------------")

    var walked = gpt.parameter_count_actual()
    print("walked total                     ", walked)
    print("formula (GPTConfig)              ", cfg.parameter_count())

    # The reconciliation: the walk of real tensors == the config formula ==
    # the published GPT-2 124M figure.
    if walked != cfg.parameter_count():
        raise Error(
            "walked total "
            + String(walked)
            + " != formula "
            + String(cfg.parameter_count())
        )
    if walked != 124_439_808:
        raise Error(
            "walked total " + String(walked) + " != published 124,439,808"
        )
    print("")
    print("Reconciled: walked == formula == 124,439,808 (GPT-2 124M).")
