#include "torch_lib.h"

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/record_function.h>


namespace torch_lib {

TorchProfilerGuard::~TorchProfilerGuard() = default;

char *from_blob(
    char *data_ptr,
    rust::Slice<const int64_t> shape,
    ScalarType dtype,
    Device device,
    rust::Box<FromBlobContext> context)
{
    // from_blob needs a copy-constructible lamba, so create a shared heap
    // reference to the supposedly uniquely-owned context box.
    auto shared_ctx = std::make_shared<rust::Box<FromBlobContext>>(std::move(context));

    // Convert the ScalarType enum from Rust to the at::ScalarType enum from C++.
    at::ScalarType scalar_type;
    switch (dtype) {
        case ScalarType::BOOL: scalar_type = at::ScalarType::Bool; break;
        case ScalarType::U8: scalar_type = at::ScalarType::Byte; break;
        case ScalarType::I8: scalar_type = at::ScalarType::Char; break;
        case ScalarType::I16: scalar_type = at::ScalarType::Short; break;
        case ScalarType::U16: scalar_type = at::ScalarType::UInt16; break;
        case ScalarType::I32: scalar_type = at::ScalarType::Int; break;
        case ScalarType::U32: scalar_type = at::ScalarType::UInt32; break;
        case ScalarType::I64: scalar_type = at::ScalarType::Long; break;
        case ScalarType::U64: scalar_type = at::ScalarType::UInt64; break;
        case ScalarType::F8_E4M3: scalar_type = at::ScalarType::Float8_e4m3fn; break;
        case ScalarType::F8_E5M2: scalar_type = at::ScalarType::Float8_e5m2; break;
        case ScalarType::F16: scalar_type = at::ScalarType::Half; break;
        case ScalarType::F32: scalar_type = at::ScalarType::Float; break;
        case ScalarType::F64: scalar_type = at::ScalarType::Double; break;
        case ScalarType::BF16: scalar_type = at::ScalarType::BFloat16; break;
        default: abort();
    }

    auto options = at::TensorOptions().dtype(scalar_type);

    // Set the device.
    switch (device.device_type) {
        case DeviceType::Cpu: options = options.device(at::kCPU); break;
        case DeviceType::Cuda: options = options.device(at::kCUDA, device.device_index); break;
        default: abort();
    }

    // Create a torch tensor.
    auto tensor = at::from_blob(
        (void *)data_ptr,
        at::IntArrayRef(shape.data(), shape.size()),
        [shared_ctx](void *) { (void)shared_ctx; },
        options
    );

    // Wrap it into a PyObject.
    return reinterpret_cast<char*>(THPVariable_Wrap(tensor));
}

ScalarType torch_to_scalar_type(char *dtype_ptr) {
    PyObject* dtype = reinterpret_cast<PyObject*>(dtype_ptr);
    switch (at::ScalarType scalar_type = ((THPDtype*)dtype)->scalar_type) {
        case at::ScalarType::Bool: return ScalarType::BOOL;
        case at::ScalarType::Char: return ScalarType::I8;
        case at::ScalarType::Byte: return ScalarType::U8;
        case at::ScalarType::Short: return ScalarType::I16;
        case at::ScalarType::UInt16: return ScalarType::U16;
        case at::ScalarType::Int: return ScalarType::I32;
        case at::ScalarType::UInt32: return ScalarType::U32;
        case at::ScalarType::Long: return ScalarType::I64;
        case at::ScalarType::UInt64: return ScalarType::U64;
        case at::ScalarType::Float8_e4m3fn: return ScalarType::F8_E4M3;
        case at::ScalarType::Float8_e5m2: return ScalarType::F8_E5M2;
        case at::ScalarType::Half: return ScalarType::F16;
        case at::ScalarType::Float: return ScalarType::F32;
        case at::ScalarType::Double: return ScalarType::F64;
        case at::ScalarType::BFloat16: return ScalarType::BF16;
        default: {
            throw std::runtime_error("Unsupported scalar type: " + std::to_string((int)scalar_type));
        }
    }
}

char *scalar_to_torch_type(ScalarType scalar_type) {
    at::ScalarType dtype;
    switch (scalar_type) {
        case ScalarType::BOOL: dtype = at::ScalarType::Bool; break;
        case ScalarType::U8: dtype = at::ScalarType::Byte; break;
        case ScalarType::I8: dtype = at::ScalarType::Char; break;
        case ScalarType::I16: dtype = at::ScalarType::Short; break;
        case ScalarType::U16: dtype = at::ScalarType::UInt16; break;
        case ScalarType::I32: dtype = at::ScalarType::Int; break;
        case ScalarType::U32: dtype = at::ScalarType::UInt32; break;
        case ScalarType::I64: dtype = at::ScalarType::Long; break;
        case ScalarType::U64: dtype = at::ScalarType::UInt64; break;
        case ScalarType::F8_E4M3: dtype = at::ScalarType::Float8_e4m3fn; break;
        case ScalarType::F8_E5M2: dtype = at::ScalarType::Float8_e5m2; break;
        case ScalarType::F16: dtype = at::ScalarType::Half; break;
        case ScalarType::F32: dtype = at::ScalarType::Float; break;
        case ScalarType::F64: dtype = at::ScalarType::Double; break;
        case ScalarType::BF16: dtype = at::ScalarType::BFloat16; break;
        default: {
            throw std::runtime_error("Unsupported scalar type: " + std::to_string((int)scalar_type));
        }
    }
    return reinterpret_cast<char*>(torch::getTHPDtype(dtype));
}

uint64_t current_stream() {
    return (int64_t)(cudaStream_t)at::cuda::getCurrentCUDAStream();
}

TorchProfilerGuard::TorchProfilerGuard(const char* name) {
    guard = std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (guard->isActive()) {
        guard->before(name);
    }
}

std::unique_ptr<TorchProfilerGuard> profile_range(rust::String name) {
    return std::make_unique<TorchProfilerGuard>(name.c_str());
}

} // namespace torch_lib
