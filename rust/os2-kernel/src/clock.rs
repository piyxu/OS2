#[derive(Debug, Default)]
pub struct DeterministicClock {
    tick: u64,
}

impl DeterministicClock {
    pub fn new() -> Self {
        Self { tick: 0 }
    }

    pub fn tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    pub fn now(&self) -> u64 {
        self.tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_is_monotonic() {
        let mut clock = DeterministicClock::new();
        let first = clock.tick();
        let second = clock.tick();
        assert!(second > first);
        assert_eq!(clock.now(), second);
    }
}
