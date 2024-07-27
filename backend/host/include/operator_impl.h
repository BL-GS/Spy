#pragma once

#include <memory>

#include "graph/op_node.h"
#include "operator/operator.h"
#include "task.h"

namespace spy::cpu {

	class CPUBackend;

	/*!
	 * @brief Execution logic of operator
	 */
	template<OperatorType T_op_type>
	struct OperatorImpl {
	public: /* Interface for schedule and execution */
		/*!
		 * @brief Get the number of task that can be executed in parallelism.
		 */
		static std::shared_ptr<ControlHeader> get_control_header([[maybe_unused]] CPUBackend *backend_ptr, [[maybe_unused]] const OperatorNode *op_node_ptr) { return nullptr; }

		static constexpr bool is_support() { return false; }

		/*!
		 * @brief Execute the operator 
		 * @param param The parameter of environment. Specifically, it denote the concurrency and thread id of CPU backend.
		 * @param op_node The OperatorNode deriving from OperatorDefinition<T_op_type>, which store necessary operands and hyper parameters.
		 * @return true if supported, otherwise false.
		 */
		static OperatorResult execute([[maybe_unused]] CPUBackend *backend_ptr, [[maybe_unused]] const OperatorEnvParam &param, [[maybe_unused]] OperatorNode *op_node_ptr) { 
            return { OperatorStatus::Unsupport, OperatorPhaseType::End };
        }
	};

#define OperatorDefinition(op_type)                                                                                                 \
    struct Operator##op_type##Impl {                                                                                                \
        using DerivedOperatorNode = OperatorDefinition<OperatorType:: op_type>;                                                     \
                                                                                                                                    \
        static std::shared_ptr<ControlHeader> get_control_header(CPUBackend *backend_ptr, const DerivedOperatorNode *op_node);             \
        static OperatorResult execute([[maybe_unused]] CPUBackend *backend_ptr, [[maybe_unused]] const OperatorEnvParam &param, [[maybe_unused]] DerivedOperatorNode *op_node); \
    };                                                                                                                              \
                                                                                                                                    \
    template<>                                                                                                                      \
    struct OperatorImpl<OperatorType:: op_type> {                                                                                   \
        using Impl                = Operator##op_type##Impl;                                                                        \
        using DerivedOperatorNode = Impl::DerivedOperatorNode;                                                                      \
                                                                                                                                    \
        static std::shared_ptr<ControlHeader> get_control_header(CPUBackend *backend_ptr, const OperatorNode *op_node) {            \
            return Impl::get_control_header(backend_ptr, static_cast<const DerivedOperatorNode *>(op_node));                        \
        }                                                                                                                           \
                                                                                                                                    \
        static constexpr bool is_support() { return true; }                                                                         \
																																	\
        static OperatorResult execute(CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {              \
            return Impl::execute(backend_ptr, param, static_cast<DerivedOperatorNode *>(op_node));                                  \
        }                                                                                                                           \
    };
    
    OperatorDefinition(Add)
    OperatorDefinition(Sub)
    OperatorDefinition(Mul)
    OperatorDefinition(Div)

    OperatorDefinition(MatMul)

    OperatorDefinition(Relu)
    OperatorDefinition(Silu)
    OperatorDefinition(Softmax)
    OperatorDefinition(NormRMS)
    OperatorDefinition(Rope)

    OperatorDefinition(Nop)
    OperatorDefinition(GetRow)
    OperatorDefinition(Dup)
    OperatorDefinition(Copy)
    OperatorDefinition(View)
    OperatorDefinition(Reshape)
    OperatorDefinition(Transpose)
    OperatorDefinition(Permute)
    OperatorDefinition(Contiguous)

    OperatorDefinition(Quantize)

#undef OperatorDefinition

} // namespace spy::cpu