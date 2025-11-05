use bytes::Bytes;
use libibverbs_sys::ibv_gid;
use serde::{Deserialize, Serialize};

use crate::{api::DomainAddress, utils::hex::fmt_hex};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Gid {
    pub raw: [u8; 16],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerbsUDAddress {
    pub gid: Gid,
    pub lid: u16,
    pub qp_num: u32,
    pub qkey: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerbsRCAddress {
    pub gid: Gid,
    pub lid: u16,
    pub qp_num: u32,
    pub psn: u32,
}

impl std::fmt::Debug for Gid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_hex(f, &self.raw)
    }
}

impl std::fmt::Display for Gid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_hex(f, &self.raw)
    }
}

impl From<Gid> for ibv_gid {
    fn from(gid: Gid) -> Self {
        Self { raw: gid.raw }
    }
}

impl VerbsUDAddress {
    const BYTES: usize = 26;
    const _SIZE_CHECK: () =
        assert!(std::mem::size_of::<VerbsUDAddress>() == Self::BYTES);

    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut bytes = [0; Self::BYTES];
        bytes[..16].copy_from_slice(&self.gid.raw);
        bytes[16..18].copy_from_slice(&self.lid.to_le_bytes());
        bytes[18..22].copy_from_slice(&self.qp_num.to_le_bytes());
        bytes[22..26].copy_from_slice(&self.qkey.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // TODO: make it more idiomatic.
        if bytes.len() != Self::BYTES {
            return None;
        }
        unsafe {
            Some(Self {
                gid: Gid { raw: bytes[..16].try_into().unwrap_unchecked() },
                lid: u16::from_le_bytes(bytes[16..18].try_into().unwrap_unchecked()),
                qp_num: u32::from_le_bytes(bytes[18..22].try_into().unwrap_unchecked()),
                qkey: u32::from_le_bytes(bytes[22..26].try_into().unwrap_unchecked()),
            })
        }
    }
}

impl From<&VerbsUDAddress> for DomainAddress {
    fn from(addr: &VerbsUDAddress) -> Self {
        DomainAddress(Bytes::copy_from_slice(&addr.to_bytes()))
    }
}
