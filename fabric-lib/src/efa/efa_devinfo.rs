use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    ptr::{NonNull, null, null_mut},
};

use libfabric_sys::{
    FI_ENOMEM, FI_EP_RDM, FI_HMEM, FI_LOCAL_COMM, FI_MR_ALLOCATED, FI_MR_HMEM,
    FI_MR_LOCAL, FI_MR_PROV_KEY, FI_MR_VIRT_ADDR, FI_MSG, FI_REMOTE_COMM, FI_RMA,
    FI_THREAD_DOMAIN, fi_dupinfo, fi_freeinfo, fi_getinfo, fi_info, make_fi_version,
};

use crate::{
    error::{LibfabricError, Result},
    provider::RdmaDomainInfo,
};

pub struct EfaDomainInfo {
    pub fi: NonNull<fi_info>,
}

unsafe impl Send for EfaDomainInfo {}
unsafe impl Sync for EfaDomainInfo {}

impl EfaDomainInfo {
    pub fn dup(info: NonNull<fi_info>) -> Self {
        // Copy fi_info. fi_dupinfo does not copy next.
        let fi =
            NonNull::new(unsafe { fi_dupinfo(info.as_ptr()) }).expect("fi_dupinfo");
        EfaDomainInfo { fi }
    }

    pub fn fi(&self) -> NonNull<fi_info> {
        self.fi
    }
}

impl RdmaDomainInfo for EfaDomainInfo {
    fn name(&self) -> Cow<'_, str> {
        unsafe {
            CStr::from_ptr((*(*self.fi.as_ptr()).domain_attr).name).to_string_lossy()
        }
    }

    fn link_speed(&self) -> u64 {
        unsafe {
            let fi = self.fi.as_ref();
            if fi.nic.is_null() {
                return 200_000_000_000;
            }
            (*(*fi.nic).link_attr).speed as u64
        }
    }
}

impl Clone for EfaDomainInfo {
    fn clone(&self) -> Self {
        EfaDomainInfo::dup(self.fi)
    }
}

impl Drop for EfaDomainInfo {
    fn drop(&mut self) {
        unsafe { fi_freeinfo(self.fi.as_ptr()) }
    }
}

pub fn get_efa_domains() -> Result<Vec<EfaDomainInfo>> {
    let mut vec = Vec::new();
    unsafe {
        let mut hints = NonNull::new(fi_dupinfo(null()))
            .ok_or_else(|| LibfabricError::new(FI_ENOMEM as i32, "fi_dupinfo"))?;
        let h = hints.as_mut();
        let provider_name = std::env::var("PPLX_GARDEN_LIBFABRIC_PROVIDER")
            .or_else(|_| std::env::var("FI_PROVIDER"))
            .unwrap_or_else(|_| "efa".to_string());
        let is_efa = provider_name == "efa";
        h.caps = if is_efa {
            FI_MSG as u64 | FI_RMA as u64 | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM
        } else {
            0
        };
        (*h.ep_attr).type_ = FI_EP_RDM;
        let provider_name = CString::new(provider_name)
            .map_err(|_| LibfabricError::new(libc::EINVAL, "provider name"))?;
        (*h.fabric_attr).prov_name = provider_name.as_ptr() as *mut libc::c_char;
        if is_efa {
            (*h.domain_attr).mr_mode = (FI_MR_LOCAL
                | FI_MR_HMEM
                | FI_MR_VIRT_ADDR
                | FI_MR_ALLOCATED
                | FI_MR_PROV_KEY) as i32;
            (*h.domain_attr).threading = FI_THREAD_DOMAIN;
        }

        let hint_ptr = if is_efa { h as *mut fi_info } else { null_mut() };
        let mut info = null_mut();
        let flags = 0;
        let ret = fi_getinfo(
            make_fi_version(1, 22),
            null(),
            null(),
            flags,
            hint_ptr,
            &mut info,
        );
        if std::env::var_os("PPLX_GARDEN_DEBUG_FI").is_some() {
            eprintln!(
                "pplx fi_getinfo provider={} is_efa={} ret={} info={:?}",
                provider_name.to_string_lossy(),
                is_efa,
                ret,
                info
            );
        }

        // Avoid fi_freeinfo freeing the CString-owned prov_name.
        (*h.fabric_attr).prov_name = null_mut();
        fi_freeinfo(h);

        let info =
            NonNull::new(info).ok_or_else(|| LibfabricError::new(ret, "fi_getinfo"))?;
        let mut fi = info;
        loop {
            if std::env::var_os("PPLX_GARDEN_DEBUG_FI").is_some() {
                let fi_ref = fi.as_ref();
                let name = CStr::from_ptr((*fi_ref.domain_attr).name).to_string_lossy();
                let provider =
                    CStr::from_ptr((*fi_ref.fabric_attr).prov_name).to_string_lossy();
                let mr_mode = (*fi_ref.domain_attr).mr_mode;
                eprintln!(
                    "pplx fi_info provider={} domain={} mr_mode=0x{:x}",
                    provider, name, mr_mode
                );
            }
            let provider =
                CStr::from_ptr((*fi.as_ref().fabric_attr).prov_name).to_string_lossy();
            if is_efa || provider == provider_name.to_string_lossy() {
                vec.push(EfaDomainInfo::dup(fi));
            }
            let Some(next) = NonNull::new((*fi.as_ptr()).next) else {
                break;
            };
            fi = next;
        }

        fi_freeinfo(info.as_ptr());
    }
    Ok(vec)
}
