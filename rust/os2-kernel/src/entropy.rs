use rand::{RngCore, SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};

#[derive(Debug)]
pub struct EntropyBalancer {
    rng: StdRng,
    counter: u64,
}

impl EntropyBalancer {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            counter: 0,
        }
    }

    /// Generate a deterministic 64-bit value mixed with a running counter.
    ///
    /// ```
    /// use os2_kernel::entropy::EntropyBalancer;
    ///
    /// let mut entropy = EntropyBalancer::new(42);
    /// let first = entropy.next_u64();
    /// let second = entropy.next_u64();
    /// assert_ne!(first, second);
    ///
    /// let mut replay = EntropyBalancer::new(42);
    /// assert_eq!(first, replay.next_u64());
    /// assert_eq!(second, replay.next_u64());
    /// ```
    pub fn next_u64(&mut self) -> u64 {
        let raw = self.rng.next_u64();
        self.counter = self.counter.wrapping_add(1);
        let mut hasher = Sha256::new();
        hasher.update(raw.to_le_bytes());
        hasher.update(self.counter.to_le_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        u64::from_le_bytes(bytes)
    }

    pub fn mix_entropy(&mut self, feedback: &[u8]) {
        if feedback.is_empty() {
            return;
        }
        let mut hasher = Sha256::new();
        hasher.update(self.counter.to_le_bytes());
        hasher.update(feedback);
        let digest = hasher.finalize();
        let mut reseed = [0u8; 32];
        reseed.copy_from_slice(&digest[..32]);
        self.rng = StdRng::from_seed(reseed);
    }
}
