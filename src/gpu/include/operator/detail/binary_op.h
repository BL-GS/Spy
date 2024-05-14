#pragma once

#include "number/tensor.h"
#include "gpu_device.h"

namespace spy::gpu {

    void cuda_op_repeat(DeviceContext & ctx,   Tensor &result, Tensor &operand_0, Tensor &operand_1);
    
    void cuda_op_add(DeviceContext & ctx,      Tensor &result, Tensor &operand_0, Tensor &operand_1);

    void cuda_op_mul(DeviceContext & ctx,      Tensor &result, Tensor &operand_0, Tensor &operand_1);

    void cuda_op_div(DeviceContext & ctx,      Tensor &result, Tensor &operand_0, Tensor &operand_1);

} // namespace spy::gpu
