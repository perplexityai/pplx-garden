use std::borrow::Cow;

use crate::{efa::EfaDomainInfo, provider::RdmaDomainInfo, verbs::VerbsDeviceInfo};

#[derive(Clone)]
pub enum DomainInfo {
    Efa(EfaDomainInfo),
    Verbs(VerbsDeviceInfo),
}

impl RdmaDomainInfo for DomainInfo {
    fn name(&self) -> Cow<'_, str> {
        match self {
            DomainInfo::Efa(info) => info.name(),
            DomainInfo::Verbs(info) => info.name(),
        }
    }

    fn link_speed(&self) -> u64 {
        match self {
            DomainInfo::Efa(info) => info.link_speed(),
            DomainInfo::Verbs(info) => info.link_speed(),
        }
    }
}
