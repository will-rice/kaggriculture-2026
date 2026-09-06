//! A bit-exact port of `CPython`'s `random.Random` for the calls the reference
//! engine makes: `random()`, `choice()` and `getrandbits()`.
//!
//! The engine seeds one generator per day with
//! `random.Random((seed * 1_000_003) ^ day)`, so the seeding path (Python
//! integer -> absolute value -> 32-bit little-endian words -> `init_by_array`)
//! has to match too.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_B0DF;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7FFF_FFFF;

#[derive(Clone)]
pub struct PyRandom {
    state: [u32; N],
    index: usize,
}

impl std::fmt::Debug for PyRandom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyRandom")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl PyRandom {
    /// `random.Random(seed)` for an integer seed of any sign.
    pub fn new(seed: i128) -> PyRandom {
        let magnitude = seed.unsigned_abs();
        let mut key = [0u32; 4];
        let mut used = 0usize;
        for (i, slot) in key.iter_mut().enumerate() {
            let word = ((magnitude >> (32 * i)) & 0xFFFF_FFFF) as u32;
            *slot = word;
            if word != 0 {
                used = i + 1;
            }
        }
        // CPython always passes at least one word (`0` seeds with key [0]).
        PyRandom::from_key(&key[..used.max(1)])
    }

    fn init_genrand(seed: u32) -> [u32; N] {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let previous = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        state
    }

    /// `init_by_array` from the reference MT19937 implementation.
    pub fn from_key(key: &[u32]) -> PyRandom {
        let mut state = Self::init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        let rounds = N.max(key.len());
        for _ in 0..rounds {
            let previous = state[i - 1];
            state[i] = (state[i] ^ ((previous ^ (previous >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                state[0] = state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let previous = state[i - 1];
            state[i] = (state[i] ^ ((previous ^ (previous >> 30)).wrapping_mul(1_566_083_941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                state[0] = state[N - 1];
                i = 1;
            }
        }
        state[0] = UPPER_MASK;
        PyRandom { state, index: N }
    }

    fn twist(&mut self) {
        let state = &mut self.state;
        for k in 0..N {
            let y = (state[k] & UPPER_MASK) | (state[(k + 1) % N] & LOWER_MASK);
            let mut value = state[(k + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                value ^= MATRIX_A;
            }
            state[k] = value;
        }
        self.index = 0;
    }

    /// `genrand_uint32`: one tempered 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^ (y >> 18)
    }

    /// `random.random()`: a 53-bit float in `[0, 1)`.
    pub fn random(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// `random.getrandbits(k)` for `0 <= k <= 64`.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return u64::from(self.next_u32() >> (32 - k));
        }
        // CPython fills 32-bit words little-endian, least significant first.
        let low = u64::from(self.next_u32());
        let high = u64::from(self.next_u32() >> (64 - k));
        low | (high << 32)
    }

    /// `random._randbelow(n)`: rejection sampling over `bit_length(n)` bits.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.choice(seq)` expressed as the chosen index.
    pub fn choice_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "cannot choose from an empty sequence");
        self.randbelow(len as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_seed_has_length_one_key() {
        // random.Random(0).getrandbits(32) == 3626764237 on every CPython 3.
        let mut rng = PyRandom::new(0);
        assert_eq!(rng.next_u32(), 3_626_764_237);
    }

    #[test]
    fn negative_seed_uses_absolute_value() {
        let mut a = PyRandom::new(-12345);
        let mut b = PyRandom::new(12345);
        assert_eq!(a.next_u32(), b.next_u32());
    }
}
