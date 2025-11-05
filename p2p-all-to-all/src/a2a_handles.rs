use fabric_lib::api::{DomainAddress, MemoryRegionDescriptor};

pub struct AllToAllRankHandle {
    pub address: DomainAddress,
    pub num_routed_desc: MemoryRegionDescriptor,
    pub recv_buffer_desc: MemoryRegionDescriptor,
}

impl AllToAllRankHandle {
    pub fn new(
        address: DomainAddress,
        num_routed_desc: MemoryRegionDescriptor,
        recv_buffer_desc: MemoryRegionDescriptor,
    ) -> Self {
        Self { address, num_routed_desc, recv_buffer_desc }
    }
}
