use libc::{CPU_SET, CPU_ZERO, cpu_set_t, pthread_self, pthread_setaffinity_np};
use syscalls::Errno;

pub fn pin_cpu(cpu: usize) -> Result<(), Errno> {
    unsafe {
        let mut cpuset = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        CPU_SET(cpu, &mut cpuset);
        let ret =
            pthread_setaffinity_np(pthread_self(), size_of::<cpu_set_t>(), &cpuset);
        if ret != 0 {
            return Err(Errno::new(ret));
        }
        Ok(())
    }
}
