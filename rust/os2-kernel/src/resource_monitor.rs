use serde::{Deserialize, Serialize};

use crate::resource::ResourceUsageSnapshot;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceObservation {
    pub timestamp: u64,
    pub usage: ResourceUsageSnapshot,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceObserver {
    history: Vec<ResourceObservation>,
}

impl ResourceObserver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, timestamp: u64, usage: ResourceUsageSnapshot) {
        self.history.push(ResourceObservation { timestamp, usage });
    }

    pub fn history(&self) -> &[ResourceObservation] {
        &self.history
    }

    pub fn latest(&self) -> Option<&ResourceObservation> {
        self.history.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_observations_tracks_latest_usage() {
        let mut observer = ResourceObserver::new();
        observer.record(1, ResourceUsageSnapshot::default());

        let mut usage = ResourceUsageSnapshot::default();
        usage.cpu.consumed = 5;
        observer.record(2, usage);

        assert_eq!(observer.history().len(), 2);
        let latest = observer.latest().expect("latest observation");
        assert_eq!(latest.timestamp, 2);
        assert_eq!(latest.usage.cpu.consumed, 5);
    }
}
