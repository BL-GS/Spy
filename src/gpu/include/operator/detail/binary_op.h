#pragma once

#include "number/tensor.h"
#include "gpu_device.h"

namespace spy::gpu {

    void cuda_op_repeat(DeviceContext & ctx,   const Tensor &result, const Tensor &operand_0, const Tensor &operand_1);
    
    void cuda_op_add(DeviceContext & ctx,      const Tensor &result, const Tensor &operand_0, const Tensor &operand_1);

    void cuda_op_sub(DeviceContext & ctx,      const Tensor &result, const Tensor &operand_0, const Tensor &operand_1);

    void cuda_op_mul(DeviceContext & ctx,      const Tensor &result, const Tensor &operand_0, const Tensor &operand_1);

    void cuda_op_div(DeviceContext & ctx,      const Tensor &result, const Tensor &operand_0, const Tensor &operand_1);

} // namespace spy::gpu
