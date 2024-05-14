#pragma once

#include "number/tensor.h"
#include "gpu_device.h"

namespace spy::gpu {

    void cuda_op_matmul(DeviceContext & ctx,      Tensor &result, Tensor &operand_0, Tensor &operand_1);

} // namespace spy::gpu