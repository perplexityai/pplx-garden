use std::collections::HashSet;

use anyhow::{Error, Result};

use crate::{
    RdmaDomainInfo, provider_dispatch::DomainInfo, topo::detect_topology,
    transfer_engine::TransferEngine, worker::Worker,
};

struct GpuDomainSpec {
    cuda_device: u8,
    domains: Vec<DomainInfo>,
    pin_worker_cpu: u16,
    pin_uvm_cpu: u16,
}

#[derive(Default)]
pub struct TransferEngineBuilder {
    gpus: Vec<GpuDomainSpec>,
}

impl TransferEngineBuilder {
    pub fn add_gpu_domains(
        &mut self,
        cuda_device: u8,
        domains: Vec<DomainInfo>,
        pin_worker_cpu: u16,
        pin_uvm_cpu: u16,
    ) {
        self.gpus.push(GpuDomainSpec {
            cuda_device,
            domains,
            pin_worker_cpu,
            pin_uvm_cpu,
        })
    }

    pub fn build(&self) -> Result<TransferEngine> {
        let system_topo = detect_topology()?;

        // Validate that there's no duplicated GPUs
        let num_gpus =
            self.gpus.iter().map(|s| s.cuda_device).collect::<HashSet<_>>().len();
        if num_gpus != self.gpus.len() {
            return Err(Error::msg("Duplicated GPUs in the builder"));
        }
        if num_gpus == 0 {
            return Err(Error::msg("No GPUs in the builder"));
        }

        // Validate builder params and prepare workers
        let mut workers = Vec::with_capacity(self.gpus.len());
        for spec in self.gpus.iter() {
            let Some(topo) =
                system_topo.iter().find(|t| t.cuda_device == spec.cuda_device)
            else {
                return Err(Error::msg(format!(
                    "cuda:{} not found in system topology",
                    spec.cuda_device
                )));
            };

            let num_domains =
                spec.domains.iter().map(|d| d.name()).collect::<HashSet<_>>().len();
            if num_domains != spec.domains.len() {
                return Err(Error::msg(format!(
                    "Duplicated domains in cuda:{}",
                    spec.cuda_device
                )));
            }

            for d in spec.domains.iter() {
                if !topo.domains.iter().any(|t| t.name() == d.name()) {
                    return Err(Error::msg(format!(
                        "Domain {} not found in the topology group of cuda:{}",
                        d.name(),
                        spec.cuda_device
                    )));
                }
            }

            for cpu in &[spec.pin_worker_cpu, spec.pin_uvm_cpu] {
                if !topo.cpus.contains(cpu) {
                    return Err(Error::msg(format!(
                        "CPU {} not found in the topology group of cuda:{}",
                        cpu, spec.cuda_device
                    )));
                }
            }

            let domain_list: Vec<_> = spec.domains.to_vec();
            let worker = Worker {
                domain_list,
                pin_worker_cpu: Some(spec.pin_worker_cpu),
                pin_uvm_cpu: Some(spec.pin_uvm_cpu),
            };
            workers.push((spec.cuda_device, worker));
        }

        // Create the transfer engine.
        Ok(TransferEngine::new(workers)?)
    }
}
