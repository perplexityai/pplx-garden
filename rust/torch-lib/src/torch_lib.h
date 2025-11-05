#pragma once

#include <memory>

#include "rust/cxx.h"

namespace at {
class RecordFunction;
}

namespace torch_lib {

class TorchProfilerGuard final {
public:
    TorchProfilerGuard(const char* name);
    ~TorchProfilerGuard();

private:
    std::unique_ptr<at::RecordFunction> guard;
};

} // namespace torch_lib

#include "torch-lib/src/lib.rs.h"

namespace torch_lib {

char *from_blob(
    char *data_ptr,
    rust::Slice<const int64_t> shape,
    ScalarType dtype,
    Device device,
    rust::Box<FromBlobContext> context
);

ScalarType torch_to_scalar_type(char *obj);
char *scalar_to_torch_type(ScalarType scalar_type);

uint64_t current_stream();

std::unique_ptr<TorchProfilerGuard> profile_range(rust::String name);

} // namespace torch_lib
