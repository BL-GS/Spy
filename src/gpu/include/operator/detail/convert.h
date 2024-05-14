#pragma once

#include <cstddef>

#include "number/number_impl/type.h"
#include "number/tensor.h"
#include "gpu_device.h"

namespace spy::gpu {

    void cuda_op_convert_raw(DeviceContext & ctx, NumberType dst_type, void *dst_ptr, NumberType src_type, void *src_ptr, size_t num);

    void cuda_op_convert(DeviceContext & ctx, Tensor &result, Tensor &operand);

} // namespace spy::gpu