#pragma once

#include "number/tensor.h"
#include "gpu_device.h"

namespace spy::gpu {

    void cuda_op_matmul(DeviceContext & ctx,      const Tensor &result, const Tensor &operand_0, const Tensor &operand_1);

} // namespace spy::gpu