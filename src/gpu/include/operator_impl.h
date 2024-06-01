#pragma once

#include "operator/type.h"
#include "task.h"

namespace spy::gpu {

    class GPUBackend;

	/*!
	 * @brief Execution logic of operator
	 */
	template<OperatorType T_op_type>
	struct OperatorImpl {
	public: /* Interface for schedule and execution */
		/*!
		 * @brief Execute the operator 
		 * @param param The parameter of environment. Specifically, it denote the concurrency and thread id of CPU backend.
		 * @return true if supported, otherwise false.
		 */
		static OperatorStatus execute([[maybe_unused]] GPUBackend *backend_ptr, [[maybe_unused]] const OperatorEnvParam &param) { return OperatorStatus::Unsupport; }
	};

#define OperatorDefinition(op_type)                                 \
    struct Operator##op_type##Impl {                                \
        static OperatorStatus execute([[maybe_unused]] GPUBackend *backend_ptr, [[maybe_unused]] const OperatorEnvParam &param); \
    };                                                              \
                                                                    \
    template<>                                                      \
    struct OperatorImpl<OperatorType:: op_type> {                   \
        static OperatorStatus execute(GPUBackend *backend_ptr, const OperatorEnvParam &param) {     \
            return Operator##op_type##Impl::execute(backend_ptr, param);                            \
        }                                                                                           \
    };
    
    OperatorDefinition(Add)
    OperatorDefinition(Sub)
    OperatorDefinition(Mul)
    OperatorDefinition(Div)

    OperatorDefinition(MatMul)

    // OperatorDefinition(Relu)
    // OperatorDefinition(Silu)
    // OperatorDefinition(Softmax)
    // OperatorDefinition(NormRMS)
    // OperatorDefinition(Rope)

    // OperatorDefinition(GetRow)
    // OperatorDefinition(Dup)
    // OperatorDefinition(Copy)
    // OperatorDefinition(Contiguous)

    // OperatorDefinition(Quantize)

#undef OperatorDefinition

} // namespace spy::cpu