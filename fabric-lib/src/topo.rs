use std::{
    collections::{HashMap, HashSet},
    ffi::CStr,
    hash::Hash,
    path::PathBuf,
    str::FromStr,
};

use cuda_lib::rt::{cudaDeviceProp, cudaGetDeviceCount, cudaGetDeviceProperties};
use once_cell::sync::Lazy;

use crate::{
    efa::{EfaDomainInfo, get_efa_domains},
    error::{FabricLibError, Result},
    provider_dispatch::DomainInfo,
    verbs::{VerbsDeviceInfo, VerbsDeviceList},
};

#[derive(Clone)]
pub struct TopologyGroup {
    pub cuda_device: u8,
    pub numa: u8,
    pub domains: Vec<DomainInfo>,
    pub cpus: Vec<u16>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
struct PciAddress {
    domain: u16,
    bus: u8,
    device: u8,
    function: u8,
}

impl PciAddress {
    fn new(domain: u16, bus: u8, device: u8, function: u8) -> Self {
        PciAddress { domain, bus, device, function }
    }

    fn get_sys_path(&self) -> String {
        format!(
            "/sys/bus/pci/devices/{:04x}:{:02x}:{:02x}.{:01x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

impl FromStr for PciAddress {
    type Err = &'static str;

    fn from_str(addr: &str) -> std::result::Result<PciAddress, &'static str> {
        if addr.len() != 12 {
            return Err("PCI address must be 12 characters long");
        }
        if !(&addr[4..5] == ":" && &addr[7..8] == ":" && &addr[10..11] == ".") {
            return Err(
                "Invalid PCI address format, expected domain:bus:device.function",
            );
        }

        let domain =
            u16::from_str_radix(&addr[0..4], 16).map_err(|_| "Invalid domain")?;
        let bus = u8::from_str_radix(&addr[5..7], 16).map_err(|_| "Invalid bus")?;
        let device =
            u8::from_str_radix(&addr[8..10], 16).map_err(|_| "Invalid device")?;
        let function =
            u8::from_str_radix(&addr[11..12], 16).map_err(|_| "Invalid function")?;

        Ok(PciAddress::new(domain, bus, device, function))
    }
}

impl From<&cudaDeviceProp> for PciAddress {
    fn from(prop: &cudaDeviceProp) -> Self {
        PciAddress {
            domain: prop.pciDomainID as u16,
            bus: prop.pciBusID as u8,
            device: prop.pciDeviceID as u8,
            function: 0,
        }
    }
}

impl From<&EfaDomainInfo> for PciAddress {
    fn from(domain: &EfaDomainInfo) -> Self {
        let fi_ref = unsafe { domain.fi().as_ref() };
        let pci = unsafe {
            fi_ref
                .nic
                .as_ref()
                .expect("NIC is null")
                .bus_attr
                .as_ref()
                .expect("Bus attribute is null")
                .attr
                .pci
        };
        PciAddress {
            domain: pci.domain_id,
            bus: pci.bus_id,
            device: pci.device_id,
            function: pci.function_id,
        }
    }
}

impl From<&VerbsDeviceInfo> for PciAddress {
    fn from(domain: &VerbsDeviceInfo) -> Self {
        let dev_path = unsafe { CStr::from_ptr((*domain.device()).dev_path.as_ptr()) }
            .to_str()
            .expect("Failed to convert dev_path to str");
        let symlink = std::fs::read_link(PathBuf::from(dev_path).join("device"))
            .expect("Failed to read verbs dev_path");
        let pci_addr = symlink
            .file_name()
            .expect("Failed to read verbs dev_path basename")
            .to_string_lossy();
        if pci_addr.len() != 12 {
            panic!("Unexpected verbs PCI address format");
        }
        PciAddress {
            domain: u16::from_str_radix(&pci_addr[0..4], 16)
                .expect("Failed to parse domain"),
            bus: u8::from_str_radix(&pci_addr[5..7], 16).expect("Failed to parse bus"),
            device: u8::from_str_radix(&pci_addr[8..10], 16)
                .expect("Failed to parse device"),
            function: u8::from_str_radix(&pci_addr[11..12], 16)
                .expect("Failed to parse function"),
        }
    }
}

impl From<&DomainInfo> for PciAddress {
    fn from(domain: &DomainInfo) -> Self {
        match domain {
            DomainInfo::Efa(info) => info.into(),
            DomainInfo::Verbs(info) => info.into(),
        }
    }
}

impl std::fmt::Display for PciAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:01x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct PciDeviceId {
    vendor: u16,
    device: u16,
}

fn get_numa_physical_cpus() -> Result<Vec<Vec<u16>>> {
    // Get all CPUs by NUMA node
    let mut numa_map = HashMap::new();
    for entry in std::fs::read_dir("/sys/devices/system/node").map_err(|_| {
        FabricLibError::Custom("Failed to read /sys/devices/system/node")
    })? {
        let entry = entry.map_err(|_| {
            FabricLibError::Custom("Failed to read /sys/devices/system/node entry")
        })?;
        if !entry.path().is_dir() {
            continue;
        }
        let path = entry.path();
        let filename = path.file_name().unwrap().to_str().unwrap();
        if !filename.starts_with("node") {
            continue;
        }
        let numa_idx = filename[4..].parse::<usize>().map_err(|_| {
            FabricLibError::Custom("Failed to parse /sys/devices/system/node entry")
        })?;
        let cpulist =
            std::fs::read_to_string(entry.path().join("cpulist")).map_err(|_| {
                FabricLibError::Custom(
                    "Failed to read /sys/devices/system/node?/cpulist",
                )
            })?;
        let cpulist = parse_comma_dash_int_list(&cpulist);
        for cpu in cpulist {
            numa_map.insert(cpu, numa_idx);
        }
    }

    // Filter out only physical CPUs
    let mut numa = Vec::new();
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
        .map_err(|_| FabricLibError::Custom("Failed to read /proc/cpuinfo"))?;
    let mut processor = 0;
    let mut physical_id = 0;
    let mut physical_cpus = HashSet::new();
    for line in cpuinfo.lines() {
        let Some(split) = line.split_once(':') else { continue };
        let key = split.0.trim();
        let value = split.1.trim();
        match key {
            "processor" => {
                processor = value
                    .parse::<u16>()
                    .map_err(|_| FabricLibError::Custom("Failed to parse processor"))?;
            }
            "physical id" => {
                physical_id = value.parse::<u16>().map_err(|_| {
                    FabricLibError::Custom("Failed to parse physical id")
                })?;
            }
            "core id" => {
                let core_id = value
                    .parse::<u16>()
                    .map_err(|_| FabricLibError::Custom("Failed to parse core id"))?;
                if physical_cpus.insert((physical_id, core_id)) {
                    let numa_idx = *numa_map
                        .get(&processor)
                        .expect("Physical ID not found in numa_map");
                    if numa_idx >= numa.len() {
                        numa.resize_with(numa_idx + 1, Vec::new);
                    }
                    numa[numa_idx].push(processor);
                }
            }
            _ => {}
        }
    }

    Ok(numa)
}

fn parse_comma_dash_int_list(s: &str) -> Vec<u16> {
    let mut result = Vec::new();

    for part in s.split(',') {
        if let Some(dash_pos) = part.find('-') {
            let start = part[..dash_pos].trim().parse::<u16>();
            let end = part[dash_pos + 1..].trim().parse::<u16>();
            if let (Ok(start), Ok(end)) = (start, end) {
                for i in start..=end {
                    result.push(i);
                }
            }
        } else if let Ok(num) = part.trim().parse::<u16>() {
            result.push(num);
        }
    }

    result
}

fn parse_0xhex_u16(s: &str) -> u16 {
    let digits = s.trim().strip_prefix("0x").expect("Not a hex string");
    u16::from_str_radix(digits, 16).expect("Failed to parse hex string as u16")
}

fn read_pci_device_id(pci_addr: &PciAddress) -> Result<PciDeviceId> {
    let sys_path = pci_addr.get_sys_path();
    let vendor = parse_0xhex_u16(
        std::fs::read_to_string(format!("{}/vendor", sys_path))
            .map_err(|_| FabricLibError::Custom("Failed to read PCI vendor"))?
            .as_str(),
    );
    let device = parse_0xhex_u16(
        std::fs::read_to_string(format!("{}/device", sys_path))
            .map_err(|_| FabricLibError::Custom("Failed to read PCI device"))?
            .as_str(),
    );
    Ok(PciDeviceId { vendor, device })
}

fn get_gpu_pci_device_id() -> Result<PciDeviceId> {
    let prop = cudaGetDeviceProperties(0)?;
    let pci_addr = PciAddress::from(&prop);
    read_pci_device_id(&pci_addr)
}

#[derive(Clone)]
struct PciProp {
    pci_addr: PciAddress,
    pci_device_id: PciDeviceId,
    numa_node: usize,
    parent_bus: PciAddress,
    branching_ancestor: PciAddress,
}

fn scan_all_pci_devices() -> Result<Vec<PciProp>> {
    let mut all_pcis = Vec::new();
    for entry in std::fs::read_dir("/sys/bus/pci/devices")
        .map_err(|_| FabricLibError::Custom("Failed to read PCI devices"))?
    {
        let entry =
            entry.map_err(|_| FabricLibError::Custom("Failed to read PCI entry"))?;
        if !entry.path().is_symlink() {
            continue;
        }
        let target = std::fs::read_link(entry.path())
            .map_err(|_| FabricLibError::Custom("Failed to read symlink"))?;
        let addr_str = target
            .file_name()
            .ok_or(FabricLibError::Custom("Failed to get file name"))?
            .to_str()
            .ok_or(FabricLibError::Custom("Failed to convert to str"))?;
        let parent_addr_str = target
            .parent()
            .ok_or(FabricLibError::Custom("Failed to get parent"))?
            .file_name()
            .ok_or(FabricLibError::Custom("Failed to get parent file name"))?
            .to_str()
            .ok_or(FabricLibError::Custom("Failed to convert parent to str"))?;
        if addr_str.len() != 12 || parent_addr_str.len() != 12 {
            continue;
        }

        let addr = addr_str
            .parse::<PciAddress>()
            .map_err(|_| FabricLibError::Custom("Failed to parse PCI address"))?;

        let mut parent_bus = parent_addr_str.parse::<PciAddress>().map_err(|_| {
            FabricLibError::Custom("Failed to parse parent PCI address")
        })?;
        parent_bus.device = 0;
        parent_bus.function = 0;

        let pci_device_id = read_pci_device_id(&addr)?;

        let numa_node_path = parent_bus.get_sys_path() + "/numa_node";
        let numa_node = std::fs::read_to_string(numa_node_path)
            .map_err(|_| FabricLibError::Custom("Failed to read NUMA node"))?
            .trim()
            .parse::<usize>();
        let Ok(numa_node) = numa_node else {
            // NOTE: /sys/bus/pci/devices/0000:00:00.0/numa_node can be -1.
            continue;
        };
        all_pcis.push(PciProp {
            pci_addr: addr,
            pci_device_id,
            numa_node,
            parent_bus,
            branching_ancestor: parent_bus,
        });
    }
    all_pcis.sort_by_key(|p| p.pci_addr);

    // Count children
    let mut children_count = HashMap::new();
    for prop in &all_pcis {
        *children_count.entry(prop.parent_bus).or_insert(0) += 1;
    }

    // Find branching ancestors
    let mut parent_bus_map = HashMap::new();
    for prop in &all_pcis {
        parent_bus_map.insert(prop.pci_addr, prop.parent_bus);
    }
    for prop in &mut all_pcis {
        // Go up if the parent only has one child.
        let mut x = prop.parent_bus;
        while let Some(&count) = children_count.get(&x) {
            if count != 1 {
                break;
            }
            if let Some(parent) = parent_bus_map.get(&x) {
                x = *parent;
            } else {
                break;
            }
        }

        // Clear device ID and function ID
        x.device = 0;
        x.function = 0;

        // Set branching ancestor
        prop.branching_ancestor = x;
    }

    Ok(all_pcis)
}

struct PciTopoGroup {
    gpu: PciProp,
    nics: Vec<PciProp>,
    cpus: Vec<u16>,
}

/// Detects the topology of the system, regardless of visibility.
/// Assuming homogeneous GPUs and homogeneous NICs.
fn detect_system_topo(
    gpu_pci_device_id: PciDeviceId,
    nic_pci_device_id: PciDeviceId,
) -> Result<Vec<PciTopoGroup>> {
    struct PciSwitchGroup<'a> {
        gpus: Vec<&'a PciProp>,
        nics: Vec<&'a PciProp>,
    }

    // Scan all PCI devices. Find all GPUs and NICs.
    let all_pcis = scan_all_pci_devices()?;
    let all_gpus: Vec<_> =
        all_pcis.iter().filter(|p| p.pci_device_id == gpu_pci_device_id).collect();
    let all_nics: Vec<_> =
        all_pcis.iter().filter(|p| p.pci_device_id == nic_pci_device_id).collect();

    // Build PciSwitchGroup
    let mut switch_groups = HashMap::new();
    for gpu in &all_gpus {
        let group = switch_groups
            .entry(gpu.branching_ancestor)
            .or_insert_with(|| PciSwitchGroup { gpus: Vec::new(), nics: Vec::new() });
        group.gpus.push(gpu);
    }
    for nic in &all_nics {
        let Some(group) = switch_groups.get_mut(&nic.branching_ancestor) else {
            continue;
        };
        group.nics.push(nic);
    }
    let mut switch_groups: Vec<_> = switch_groups.into_values().collect();
    switch_groups.sort_by_key(|g| g.gpus[0].pci_addr);

    // Get NUMA physical CPUs
    let numa_cpus = get_numa_physical_cpus()?;

    // Count GPUs per NUMA node
    let mut numa_gpu_count = vec![0; numa_cpus.len()];
    for gpu in all_gpus {
        numa_gpu_count[gpu.numa_node] += 1;
    }

    // Create topology groups
    let mut system_topo = Vec::new();
    let mut numa_gpu_indices = vec![0; numa_cpus.len()];
    for switch in switch_groups.into_iter() {
        let nics_per_gpu = switch.nics.len() / switch.gpus.len();
        for (i_gpu, gpu) in switch.gpus.iter().enumerate() {
            // Assign NICs to GPUs
            let nics = &switch.nics[i_gpu * nics_per_gpu..(i_gpu + 1) * nics_per_gpu];

            // Assign CPUs to GPUs
            let numa_gpu_index = numa_gpu_indices[gpu.numa_node];
            numa_gpu_indices[gpu.numa_node] += 1;
            let cpus_per_gpu =
                numa_cpus[gpu.numa_node].len() / numa_gpu_count[gpu.numa_node];
            let cpu_start = numa_gpu_index * cpus_per_gpu;
            let cpus = &numa_cpus[gpu.numa_node][cpu_start..cpu_start + cpus_per_gpu];
            system_topo.push(PciTopoGroup {
                gpu: (*gpu).clone(),
                nics: nics.iter().map(|x| (*x).clone()).collect(),
                cpus: cpus.to_vec(),
            });
        }
    }

    Ok(system_topo)
}

fn get_visible_domains() -> Vec<DomainInfo> {
    // Try EFA first
    let efa_domains = get_efa_domains().unwrap_or_default();
    if !efa_domains.is_empty() {
        return efa_domains.into_iter().map(DomainInfo::Efa).collect();
    }

    // Then try Verbs
    let Ok(verbs_domains) = VerbsDeviceList::get_all_devices() else {
        return Vec::new();
    };
    (0..verbs_domains.num_devices)
        .map(|i| DomainInfo::Verbs(VerbsDeviceInfo::new(verbs_domains.clone(), i)))
        .collect()
}

fn do_detect_topology() -> Result<Vec<TopologyGroup>> {
    // Get the number of visible GPUs and the GPU PCI device ID
    let num_visible_gpus = cudaGetDeviceCount()? as usize;
    if num_visible_gpus == 0 {
        return Err(FabricLibError::Custom("No visible GPUs"));
    }
    let gpu_pci_device_id = get_gpu_pci_device_id()?;

    // Get visible NICs by EFA and Verbs
    let domains = get_visible_domains();
    if domains.is_empty() {
        return Err(FabricLibError::Custom("No visible NICs"));
    }

    // Get NIC PCI device ID
    let mut nic_pci_device_id_count = HashMap::new();
    for info in &domains {
        let pci_device_id = read_pci_device_id(&PciAddress::from(info))?;
        *nic_pci_device_id_count.entry(pci_device_id).or_insert(0) += 1;
    }
    // Some systems have one ConnectX-7 per GPU, but also a few additional ConnectX-6.
    // Use the most popular NIC PCI device ID.
    let nic_pci_device_id = nic_pci_device_id_count
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(pci_device_id, _)| *pci_device_id)
        .ok_or(FabricLibError::Custom("No visible NICs"))?;

    // Detect system topology
    let system_topo = detect_system_topo(gpu_pci_device_id, nic_pci_device_id)?;

    // Only keep visible NICs and GPUs (e.g., due to containerization)
    let mut visible_nics: HashMap<_, _> =
        domains.iter().map(|info| (PciAddress::from(info), info.clone())).collect();
    let mut visible_gpus = HashMap::new();
    for cuda_device in 0..num_visible_gpus {
        let prop = cudaGetDeviceProperties(cuda_device as i32)?;
        let pci_addr = PciAddress::from(&prop);
        visible_gpus.insert(pci_addr, cuda_device);
    }
    let mut topo_groups = Vec::new();
    for group in system_topo {
        let Some(cuda_device) = visible_gpus.remove(&group.gpu.pci_addr) else {
            continue;
        };

        let mut domains = Vec::new();
        for nic in &group.nics {
            if let Some(domain) = visible_nics.remove(&nic.pci_addr) {
                domains.push(domain);
            }
        }
        if domains.is_empty() {
            continue;
        }

        topo_groups.push(TopologyGroup {
            cuda_device: cuda_device as u8,
            numa: group.gpu.numa_node as u8,
            domains,
            cpus: group.cpus,
        });
    }

    Ok(topo_groups)
}

static GLOBAL: Lazy<Result<Vec<TopologyGroup>>> = Lazy::new(do_detect_topology);

pub fn detect_topology() -> Result<&'static [TopologyGroup]> {
    match Lazy::force(&GLOBAL) {
        Ok(topo) => Ok(topo),
        Err(e) => Err(e.clone()),
    }
}
