use std::{
    collections::HashMap,
    num::NonZeroU32,
    sync::{
        Arc,
        atomic::{AtomicI64, Ordering::Relaxed},
    },
};

use cuda_lib::gdr::GdrFlag;
use parking_lot::RwLock;

use crate::api::{GdrCounter, ImmCounter};

pub enum ImmCount {
    Expected { counter: Arc<AtomicI64>, expected: NonZeroU32 },
    Imm { counter: Arc<AtomicI64> },
    Gdr { counter: Arc<AtomicI64>, flag: Arc<GdrFlag> },
}

impl ImmCount {
    /// Consume self and return the current value and the expected value.
    pub fn consume(self) -> (u32, Option<NonZeroU32>) {
        match self {
            ImmCount::Expected { counter, expected } => {
                (counter.load(Relaxed) as u32, Some(expected))
            }
            ImmCount::Imm { counter } => (counter.load(Relaxed) as u32, None),
            ImmCount::Gdr { counter, .. } => (counter.load(Relaxed) as u32, None),
        }
    }

    /// Returns true if the counter has reached the exact expected value.
    /// When reached, the counter is subtracted by the expected value (likely reset to 0).
    pub fn inc(&self) -> bool {
        match &self {
            ImmCount::Expected { counter, expected } => {
                let prev = counter.fetch_add(1, Relaxed);
                let reached = prev as u32 + 1 == expected.get();
                if reached {
                    counter.fetch_sub(expected.get() as i64, Relaxed);
                }
                reached
            }
            ImmCount::Imm { counter } => {
                counter.fetch_add(1, Relaxed);
                false
            }
            ImmCount::Gdr { counter, flag } => {
                let value = counter.fetch_add(1, Relaxed) + 1;
                if value == 0 {
                    flag.set(true);
                    counter.store(0, Relaxed);
                }
                false
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImmCountStatus {
    Vacant,
    NotReached,
    Reached,
}

pub struct ImmCountMap {
    map: RwLock<HashMap<u32, ImmCount>>,
}

impl ImmCountMap {
    pub fn new() -> Self {
        Self { map: RwLock::new(HashMap::new()) }
    }

    /// Use the imm as a counter. Reset the counter to 0.
    /// Returns the previous counter if any.
    pub fn set_expected(&self, imm: u32, expected: NonZeroU32) -> Option<ImmCount> {
        self.map.write().insert(
            imm,
            ImmCount::Expected { counter: Arc::new(AtomicI64::new(0)), expected },
        )
    }

    /// Return an exposed imm counter.
    pub fn get_imm_counter(&self, imm: u32) -> ImmCounter {
        let counter = Arc::new(AtomicI64::new(0));
        let imm_counter = ImmCounter::new(counter.clone());
        self.map.write().insert(imm, ImmCount::Imm { counter });
        imm_counter
    }

    /// Return an exposed gdr counter.
    pub fn get_gdr_counter(&self, imm: u32, flag: Arc<GdrFlag>) -> GdrCounter {
        let counter = Arc::new(AtomicI64::new(0));
        let imm_counter = GdrCounter::new(counter.clone(), flag.clone());
        self.map.write().insert(imm, ImmCount::Gdr { counter, flag });
        imm_counter
    }

    /// Stop treating the imm as a counter.
    /// Return the previous counter if any.
    pub fn remove(&self, imm: u32) -> Option<ImmCount> {
        self.map.write().remove(&imm)
    }

    /// Get the current value of the counter.
    /// If the imm is not used as a counter, returns None.
    pub fn get(&self, imm: u32) -> Option<u32> {
        self.map.read().get(&imm).map(|v| match v {
            ImmCount::Expected { counter, .. } => counter.load(Relaxed) as u32,
            ImmCount::Imm { counter } => counter.load(Relaxed) as u32,
            ImmCount::Gdr { counter, .. } => counter.load(Relaxed) as u32,
        })
    }

    /// Get the expected value of the counter.
    /// If the imm is not used as a counter, returns None.
    pub fn get_expected(&self, imm: u32) -> Option<NonZeroU32> {
        let counters = self.map.read();
        match counters.get(&imm)? {
            ImmCount::Expected { expected, .. } => Some(*expected),
            ImmCount::Gdr { .. } => None,
            ImmCount::Imm { .. } => None,
        }
    }

    /// Increment the counter.
    /// Returns the status of the counter.
    /// If the imm is not used as a counter, returns Vacant.
    /// When reached, the counter is subtracted by the expected value (likely reset to 0).
    pub fn inc(&self, imm: u32) -> ImmCountStatus {
        if let Some(imm_count) = self.map.read().get(&imm) {
            if imm_count.inc() {
                ImmCountStatus::Reached
            } else {
                ImmCountStatus::NotReached
            }
        } else {
            ImmCountStatus::Vacant
        }
    }
}

impl Default for ImmCountMap {
    fn default() -> Self {
        Self::new()
    }
}
