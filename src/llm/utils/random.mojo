# Deterministic pseudo-random generator (linear congruential).
#
# Everything "random" in this project is really *seeded and reproducible*: the
# same seed must always replay the same batch order (and, later, the same weight
# initialization), so that a training run is a repeatable experiment and a test
# can assert on exact values. A linear congruential generator (LCG) is the
# smallest thing that delivers that — one 64-bit state, one multiply-add per
# draw, no tables, no external entropy.
#
# The recurrence is  state <- state * A + C  (mod 2**64), with the wraparound
# supplied for free by UInt64 two's-complement overflow. A and C are Knuth's
# MMIX constants (D. E. Knuth, TAOCP vol. 2), a well-tested full-period choice —
# not magic numbers, but a named, citable pair. An LCG is *not* cryptographic and
# its low bits are weak; that is fine here, where the only requirement is a
# reproducible shuffle, never unpredictability.
#
# The dataset pipeline needed only the integer surface (next_u64, next_below,
# shuffle). The model-parameter layer adds float draws on the *same* generator:
# uniform() in [0, 1), uniform_range(), and normal() via Box-Muller. These are
# additive — the seed semantics and integer methods are unchanged, so Part VI's
# frozen goldens still hold.

from std.math import sqrt, log, cos, pi

comptime LCG_MULTIPLIER: UInt64 = 6364136223846793005  # Knuth MMIX multiplier
comptime LCG_INCREMENT: UInt64 = 1442695040888963407  # Knuth MMIX increment


struct Rng(Copyable, Movable):
    var state: UInt64  # full 64-bit generator state

    def __init__(out self, seed: UInt64):
        # Seed the generator. Any seed is valid, including 0: an LCG with an odd
        # increment C has full period from every starting state, so seed 0 is not
        # a degenerate fixed point.
        self.state = seed

    def next_u64(mut self) -> UInt64:
        # Advance the state one step and return it. The multiply and add wrap
        # modulo 2**64 via UInt64 overflow (verified against a Python oracle), so
        # no explicit masking is needed.
        self.state = self.state * LCG_MULTIPLIER + LCG_INCREMENT
        return self.state

    def next_below(mut self, n: Int) raises -> Int:
        # Return a value in [0, n), near-uniform (see the bias note below). Raises
        # if n <= 0 (an empty range has no valid draw).
        #
        # Uses a plain modulo, which introduces a bias toward smaller results
        # because 2**64 is not an exact multiple of n. The bias magnitude is at
        # most n / 2**64; for every n this project uses (window counts up to a
        # few million) that is below 1e-13 and utterly unobservable, so the
        # rejection loop that would remove it is deliberately omitted to keep the
        # generator legible. Documented rather than hidden.
        if n <= 0:
            raise Error("Rng.next_below: n must be positive, got " + String(n))
        return Int(self.next_u64() % UInt64(n))

    def shuffle(mut self, mut items: List[Int]):
        # Shuffle `items` in place with the Fisher-Yates algorithm: walk from the
        # last index down to 1, swapping each element with a chosen one at or
        # before it. With a perfectly uniform bounded draw this visits every
        # permutation with equal probability; here it inherits next_below's
        # negligible modulo bias (documented there). Touches each element once;
        # mutates the argument, allocates nothing.
        for i in range(len(items) - 1, 0, -1):
            # j uniform in [0, i]; next_below(i + 1) cannot raise here since
            # i + 1 >= 2 > 0.
            var j = Int(self.next_u64() % UInt64(i + 1))
            var tmp = items[i]
            items[i] = items[j]
            items[j] = tmp

    def uniform(mut self) -> Float64:
        # A Float64 in [0, 1). Take the top 53 bits of the next state as the
        # mantissa (2**53 is the largest integer Float64 represents exactly), so
        # every representable value in [0, 1) is reachable with equal spacing.
        # The low bits of an LCG are the weak ones, so using the high bits is
        # deliberate, not incidental.
        var bits = self.next_u64() >> 11
        return Float64(bits) * (1.0 / 9007199254740992.0)  # / 2**53

    def uniform_range(mut self, low: Float64, high: Float64) -> Float64:
        # A Float64 in [low, high) by affine mapping of uniform(). Callers that
        # need low < high enforce it; a reversed range simply mirrors.
        return low + (high - low) * self.uniform()

    def normal(mut self, mean: Float64, std: Float64) -> Float64:
        # One standard-normal draw scaled to (mean, std) via the Box-Muller
        # transform: two uniforms -> one normal. u1 is clamped away from 0 to
        # keep log(u1) finite (the u1 = 0 case has probability zero but is not
        # impossible with a finite generator). Consumes two draws per call.
        var u1 = self.uniform()
        var u2 = self.uniform()
        if u1 < 1e-300:
            u1 = 1e-300  # guard log(0)
        var z = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
        return mean + std * z
