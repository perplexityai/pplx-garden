use std::{borrow::Cow, ffi::CStr, sync::Arc};

use crate::{
    error::{Result, VerbsError},
    provider::RdmaDomainInfo,
};

use libibverbs_sys::{ibv_device, ibv_free_device_list, ibv_get_device_list};

pub struct VerbsDeviceList {
    pub list: *mut *mut ibv_device,
    pub num_devices: usize,
}

unsafe impl Send for VerbsDeviceList {}
unsafe impl Sync for VerbsDeviceList {}

impl VerbsDeviceList {
    pub fn get_all_devices() -> Result<Arc<Self>> {
        let mut num_devices = 0;
        let list = unsafe { ibv_get_device_list(&raw mut num_devices) };
        if list.is_null() {
            Err(VerbsError::with_last_os_error("ibv_get_device_list").into())
        } else {
            Ok(Arc::new(Self { list, num_devices: num_devices as usize }))
        }
    }
}

impl Drop for VerbsDeviceList {
    fn drop(&mut self) {
        unsafe { ibv_free_device_list(self.list) };
    }
}

#[derive(Clone)]
pub struct VerbsDeviceInfo {
    pub device_list: Arc<VerbsDeviceList>,
    pub device_index: usize,
    pub port_num: u8,
    pub gid_index: u8,
}

impl VerbsDeviceInfo {
    pub fn new(device_list: Arc<VerbsDeviceList>, device_index: usize) -> Self {
        // TODO: port_num
        // TODO: gid_index
        Self { device_list, device_index, port_num: 1, gid_index: 0 }
    }

    pub fn device(&self) -> *mut ibv_device {
        unsafe { *self.device_list.list.add(self.device_index) }
    }
}

impl RdmaDomainInfo for VerbsDeviceInfo {
    fn name(&self) -> Cow<'_, str> {
        unsafe { CStr::from_ptr((*self.device()).name.as_ptr()).to_string_lossy() }
    }

    fn link_speed(&self) -> u64 {
        let path = format!(
            "{}/ports/{}/rate",
            unsafe {
                CStr::from_ptr((*self.device()).ibdev_path.as_ptr()).to_string_lossy()
            },
            self.port_num
        );
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let trimmed = content.trim();
                let end_pos = trimmed.find(' ').unwrap_or(trimmed.len());
                let gbps: f64 = trimmed[..end_pos].parse().unwrap_or(0.0);
                (gbps * 1e9) as u64
            }
            Err(_) => 0,
        }
    }
}
