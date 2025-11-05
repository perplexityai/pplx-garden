#pragma once

#include <cstdint>

namespace rose {

__forceinline__ __device__ void st_volatile_u32(uint32_t *flag_addr, uint32_t flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

__forceinline__ __device__ uint32_t ld_volatile_u32(uint32_t *flag_addr) {
  uint32_t flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

__forceinline__ __device__ uint32_t ld_acquire_u32(uint32_t *flag_addr) {
  uint32_t flag;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

__forceinline__ __device__ uint8_t ld_mmio_b8(uint8_t *flag_addr) {
  uint32_t tmp;
  asm volatile(
    "{ ld.mmio.relaxed.sys.global.b8 %0, [%1]; }"
    : "=r"(tmp)
    : "l"(flag_addr)
    :
  );
  return static_cast<uint8_t>(tmp);
}

__forceinline__ __device__ void st_mmio_b8(uint8_t *flag_addr, uint8_t flag) {
  uint32_t tmp = static_cast<uint32_t>(flag);
  asm volatile(
    "{ st.mmio.relaxed.sys.global.b8 [%1], %0; }"
    :
    : "r"(tmp), "l"(flag_addr)
    :
  );
}

__forceinline__ __device__ void st_release_u32(uint32_t *flag_addr, uint32_t flag) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
}

__forceinline__ __device__ void st_relaxed_u32(uint32_t *flag_addr, uint32_t flag) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
}

__forceinline__ __device__ uint32_t add_release_sys_u32(uint32_t *addr, uint32_t val) {
  uint32_t flag;
  asm volatile("atom.release.sys.global.add.u32 %0, [%1], %2;" : "=r"(flag) : "l"(addr), "r"(val));
  return flag;
}

__forceinline__ __device__ uint32_t add_release_gpu_u32(uint32_t *addr, uint32_t val) {
  uint32_t flag;
  asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(flag) : "l"(addr), "r"(val));
  return flag;
}

__forceinline__ __device__ void fence_acq_rel_gpu() {
    asm volatile("{ fence.acq_rel.gpu; }":: : "memory");
}

__forceinline__ __device__ void fence_acquire_gpu() {
    asm volatile("{ fence.acquire.gpu; }":: : "memory");
}

__forceinline__ __device__ void fence_release_gpu() {
    asm volatile("{ fence.release.gpu; }":: : "memory");
}

__forceinline__ __device__ void fence_acquire_system() {
    asm volatile("{ fence.acquire.sys; }":: : "memory");
}

__forceinline__ __device__ void fence_release_system() {
    asm volatile("{ fence.release.sys; }":: : "memory");
}

__forceinline__ __device__ uint4 ld_global_uint4(const void *ptr) {
  uint4 v;
  asm volatile(
      "{ ld.global.v4.u32 {%0, %1, %2, %3}, [%4]; }"
      : "=r"(v.x)
      , "=r"(v.y)
      , "=r"(v.z)
      , "=r"(v.w)
      : "l"(ptr)
  );
  return v;
}

__forceinline__ __device__ void st_global_uint4(void *ptr, uint4 v) {
  asm volatile(
      "{ st.global.v4.u32 [%0], {%1, %2, %3, %4}; }"
      :
      : "l"(ptr)
      , "r"(v.x)
      , "r"(v.y)
      , "r"(v.z)
      , "r"(v.w)
  );
}

__forceinline__ __device__ uint4 ld_global_nc_uint4(const void *ptr) {
  uint4 v;
  asm volatile(
      "{ ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4]; }"
      : "=r"(v.x)
      , "=r"(v.y)
      , "=r"(v.z)
      , "=r"(v.w)
      : "l"(ptr)
  );
  return v;
}

__forceinline__ __device__ void st_global_nc_uint4(void *ptr, uint4 v) {
  asm volatile(
      "{ st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4}; }"
      :
      : "l"(ptr)
      , "r"(v.x)
      , "r"(v.y)
      , "r"(v.z)
      , "r"(v.w)
  );
}

__forceinline__ __device__ uint4 ld_shared_uint4(const void *ptr) {
  uint4 v;
  asm volatile(
      "{ ld.shared.v4.u32 {%0, %1, %2, %3}, [%4]; }"
      : "=r"(v.x)
      , "=r"(v.y)
      , "=r"(v.z)
      , "=r"(v.w)
      : "l"(ptr)
  );
  return v;
}

__forceinline__ __device__ void st_shared_uint4(void *ptr, uint4 v) {
  asm volatile(
      "{ st.shared.v4.u32 [%0], {%1, %2, %3, %4}; }"
      :
      : "l"(ptr)
      , "r"(v.x)
      , "r"(v.y)
      , "r"(v.z)
      , "r"(v.w)
  );
}

} // namespace rose
