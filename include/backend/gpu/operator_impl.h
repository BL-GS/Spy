#pragma once

#include "operator/type.h"
#include "operator/config.h"
#include "graph/graph.h"

namespace spy {
	class GPUBackend;
}

namespace spy::gpu {

	/*!
	 * @brief Execution logic of operator
	 */
	template<OperatorType T_op_type>
	struct OperatorImpl {
	public: /* Interface for schedule and execution */
		/*!
		 * @brief Get the number of task that can be executed in parallelism.
		 */
		static size_t get_task_num([[maybe_unused]] const GPUBackend *backend_ptr, [[maybe_unused]] const OperatorNode *op_node) { return 1; }

		/*!
		 * @brief Get the size of buffer that can be executed in parallelism.
		 * @return 0 if buffer is not necessary
		 */
		static size_t get_buffer_size([[maybe_unused]] const GPUBackend *backend_ptr, [[maybe_unused]] const OperatorNode *op_node) { return 0; }

		/*!
		 * @brief Execute the operator 
		 * @param param The parameter of environment. Specifically, it denote the concurrency and thread id of CPU backend.
		 * @param op_node The OperatorNode deriving from OperatorDefinition<T_op_type>, which store necessary operands and hyper parameters.
		 * @return true if supported, otherwise false.
		 */
		static bool execute([[maybe_unused]] GPUBackend *backend_ptr, [[maybe_unused]] const OperatorEnvParam &param, [[maybe_unused]] OperatorNode *op_node) { return false; }
	};

#define OperatorDefinition(op_type)                                 \
    struct Operator##op_type##Impl {                                \
        static size_t get_task_num([[maybe_unused]] const GPUBackend *backend_ptr, const OperatorNode *op_node);        \
        static size_t get_buffer_size([[maybe_unused]] const GPUBackend *backend_ptr, const OperatorNode *op_node);     \
        static bool   execute([[maybe_unused]] GPUBackend *backend_ptr, [[maybe_unused]] const OperatorEnvParam &param, [[maybe_unused]] OperatorNode *op_node); \
    };                                                              \
                                                                    \
    template<>                                                      \
    struct OperatorImpl<OperatorType:: op_type> {                   \
        static size_t get_task_num([[maybe_unused]] const GPUBackend *backend_ptr, const OperatorNode *op_node) { \
            return Operator##op_type##Impl::get_task_num(backend_ptr, op_node);                             \
        }                                                                                                   \
                                                                                                            \
        static size_t get_buffer_size([[maybe_unused]] const GPUBackend *backend_ptr, const OperatorNode *op_node) { \
            return Operator##op_type##Impl::get_buffer_size(backend_ptr, op_node);                          \
        }                                                                                                   \
                                                                                                            \
        static bool execute([[maybe_unused]] GPUBackend *backend_ptr, [[maybe_unused]] const OperatorEnvParam &param, [[maybe_unused]] OperatorNode *op_node) { \
            return Operator##op_type##Impl::execute(backend_ptr, param, op_node);                           \
        }                                                                                                   \
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