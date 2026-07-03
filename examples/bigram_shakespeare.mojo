# Train a bigram language model on tiny Shakespeare and generate from it.
#
# Puts the whole Part VII surface to work end to end: a TrainingConfig, the
# character tokenizer, the batch loader, the bigram model's fused loss+gradient,
# an SGD step, and seeded categorical sampling. It trains over the corpus for a
# fixed step budget, printing loss and perplexity periodically, then generates a
# few hundred characters starting from a newline.
#
# Run:
#     pixi run mojo run -I src examples/bigram_shakespeare.mojo
#
# The output is bigram-plausible, not fluent: a bigram only knows the current
# character, so it reproduces letter-pair statistics (a 'q' is almost always
# followed by 'u') but has no memory beyond one step. That limitation is the
# whole motivation for the Transformer that follows.

from std.math import exp

from llm.config import TrainingConfig
from llm.tokenizer.char import CharTokenizer
from llm.data.corpus import load_text
from llm.data.dataset import TokenDataset, train_val_split
from llm.data.loader import BatchLoader
from llm.models.bigram import BigramLM
from llm.tensor.tensor2d import zeros_2d
from llm.training.optimizer import sgd_step
from llm.training.loss import perplexity
from llm.generation.sampler import sample_categorical
from llm.utils.random import Rng


def generate(
    model: BigramLM,
    start_id: Int,
    count: Int,
    temperature: Float64,
    mut rng: Rng,
) raises -> List[Int]:
    # Autoregressive sampling: from `start_id`, repeatedly turn the current
    # token's next-token distribution into a draw and feed it back. The generate
    # *loop* lives here in the example rather than the library — the real,
    # reusable generate (with top-k/top-p) arrives with the generation chapter.
    var out: List[Int] = []
    var current = start_id
    for _ in range(count):
        var probs = model.next_probs(current, temperature)
        var nxt = sample_categorical(probs, rng)
        out.append(nxt)
        current = nxt
    return out^


def main() raises:
    var config = TrainingConfig(
        batch_size=32,
        learning_rate=1.0,
        max_steps=3000,
        seed=42,
    )
    config.validate()

    var text = load_text("data/tinyshakespeare/input.txt")
    var tokenizer = CharTokenizer.from_text(text)
    var vocab_size = tokenizer.vocab_size()
    print("corpus:", text.byte_length(), "bytes, vocab:", vocab_size, "chars")

    var ids = tokenizer.encode(text)
    var split = train_val_split(ids, 0.1)
    var loader = BatchLoader(
        split.train.copy(), config.batch_size, seq_len=16, stride=16
    )

    var model = BigramLM(vocab_size)
    var grad = zeros_2d(vocab_size, vocab_size)

    # A uniform model starts at loss log(V); watch it fall toward the corpus's
    # bigram entropy.
    var epoch: UInt64 = 0
    loader.start_epoch(config.seed + epoch)
    for step in range(config.max_steps):
        if not loader.has_next():
            epoch += 1
            loader.start_epoch(config.seed + epoch)
        var batch = loader.next_batch()
        var loss = model.loss_and_grad(batch, grad)
        sgd_step(model.table, grad, config.learning_rate)
        if step % 500 == 0:
            print(
                "step",
                step,
                "loss",
                loss,
                "perplexity",
                perplexity(loss),
            )

    # Generate ~300 characters starting from a newline, temperature 1.0, seeded.
    var newline_id = tokenizer.encode("\n")[0]
    var sampler_rng = Rng(config.seed)
    var generated = generate(model, newline_id, 300, 1.0, sampler_rng)
    print("\n--- sample ---")
    print(tokenizer.decode(generated))
