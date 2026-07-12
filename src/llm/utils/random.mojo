"""Deterministic pseudo-random generator (linear congruential).

Everything "random" here is seeded and reproducible: the same seed replays the
same batch order and weight initialization, so a run is a repeatable experiment
and a test can assert exact values. The recurrence is state <- state * A + C
(mod 2**64), with A and C being Knuth's MMIX constants (TAOCP vol. 2), a
full-period choice. An LCG is not cryptographic and its low bits are weak, which
is fine when the only requirement is a reproducible shuffle.
"""

from std.math import sqrt, log, cos, pi

comptime LCG_MULTIPLIER: UInt64 = 6364136223846793005  # Knuth MMIX multiplier
comptime LCG_INCREMENT: UInt64 = 1442695040888963407  # Knuth MMIX increment

# 2**-53: the spacing of the [0, 1) grid uniform() draws on. 2**53 is the
# largest integer Float64 represents exactly, so scaling a 53-bit integer by
# this exact power of two is bit-exact — the derivation is the documentation,
# replacing a bare mantissa literal. 1 << 53 is an exact IntLiteral before the
# Float64 cast, so no precision is lost forming the constant.
comptime INV_2_POW_53 = 1.0 / Float64(1 << 53)

# One full turn in radians, for the Box-Muller angle 2*pi*u2.
comptime TWO_PI = 2.0 * pi


struct Rng(Copyable, Movable):
    """A seeded 64-bit linear congruential generator."""

    var state: UInt64  # full 64-bit generator state

    def __init__(out self, seed: UInt64):
        """Seed the generator.

        Any seed is valid, including 0: an LCG with an odd increment has full
        period from every starting state.
        """
        self.state = seed

    def next_u64(mut self) -> UInt64:
        """Advance the state one step and return it.

        The multiply and add wrap modulo 2**64 via UInt64 overflow, so no
        explicit masking is needed.
        """
        self.state = self.state * LCG_MULTIPLIER + LCG_INCREMENT
        return self.state

    def next_below(mut self, n: Int) raises -> Int:
        """Return a near-uniform value in [0, n).

        Uses a plain modulo, which biases toward smaller results because 2**64 is
        not an exact multiple of n. The bias is at most n / 2**64 (below 1e-13
        for every n used here), so the rejection loop is omitted for legibility.

        Raises:
            Error: If n is not positive.
        """
        if n <= 0:
            raise Error("Rng.next_below: n must be positive, got " + String(n))
        return Int(self.next_u64() % UInt64(n))

    def shuffle(mut self, mut items: List[Int]):
        """Shuffle `items` in place with the Fisher-Yates algorithm.

        Walk from the last index down to 1, swapping each element with a chosen
        one at or before it. Inherits next_below's negligible modulo bias.
        Mutates the argument; allocates nothing.
        """
        for i in range(len(items) - 1, 0, -1):
            # j uniform in [0, i]; next_below(i + 1) cannot raise here since
            # i + 1 >= 2 > 0.
            var j = Int(self.next_u64() % UInt64(i + 1))
            var tmp = items[i]
            items[i] = items[j]
            items[j] = tmp

    def uniform(mut self) -> Float64:
        """Return a Float64 in [0, 1).

        Takes the top 53 bits of the next state as the mantissa (2**53 is the
        largest integer Float64 represents exactly), so every representable value
        is reachable with equal spacing. Using the high bits avoids an LCG's weak
        low bits.
        """
        var bits = self.next_u64() >> 11
        return Float64(bits) * INV_2_POW_53

    def uniform_range(mut self, low: Float64, high: Float64) -> Float64:
        """Return a Float64 in [low, high) by affine mapping of uniform()."""
        return low + (high - low) * self.uniform()

    def normal(mut self, mean: Float64, std: Float64) -> Float64:
        """Return one normal draw with the given mean and std via Box-Muller.

        Two uniforms map to one normal. u1 is clamped away from 0 to keep
        log(u1) finite. Consumes two draws per call.
        """
        var u1 = self.uniform()
        var u2 = self.uniform()
        if u1 < 1e-300:
            u1 = 1e-300  # guard log(0)
        var z = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2)
        return mean + std * z
