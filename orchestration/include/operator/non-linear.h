#pragma once

#include "util/shell/logger.h"
#include "operator/type.h"
#include "operator/config.h"
#include "graph/graph.h"

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Relu> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Relu;

	public:
		OperatorDefinition(): OperatorNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            spy_assert(num_input() == 1, "Expect the number of operands to be 1 (cur: {})", num_input());

			const Tensor &operand_0 = input(0).tensor;

			spy_assert(operand_0.get_number_type() == NumberType::FP32);

			const auto &shape_0   = operand_0.get_shape();
			const Shape shape_res = shape_0;
			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::Silu> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Silu;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            spy_assert(num_input() == 1, "Expect the number of operands to be 1 (cur: {})", num_input());

			const Tensor &operand_0 = input(0).tensor;

			spy_assert(operand_0.get_number_type() == NumberType::FP32);

			const auto &shape_0   = operand_0.get_shape();
			const Shape shape_res = shape_0;
			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::Softmax> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Softmax;

	public:
		float scale;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

		OperatorDefinition(const float scale): OperatorNode(TYPE), scale(scale) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			spy_assert(num_input() == 1 || num_input() == 2, "Expect the number of operands to be 1 or 2 (cur: {})", num_input());

			const Tensor &operand_0 = input(0).tensor;
			const auto &shape_0     = operand_0.get_shape();
			const Shape shape_res   = shape_0;
			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::NormRMS> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::NormRMS;

    public:
        float eps;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

		OperatorDefinition(float eps): OperatorNode(TYPE), eps(eps) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			spy_assert(num_input() == 1, "Expect the number of operands to be 1 (cur: {})", num_input());
			spy_assert(eps > 0.0F, "Expect the eps > 0.0 (cur: {})", eps);

			const Tensor &operand_0 = input(0).tensor;
			const auto &shape_0     = operand_0.get_shape();
			const Shape shape_res   = shape_0;
			return { shape_res, nullptr };
		}
    };

    template<>
    struct OperatorDefinition<OperatorType::Rope> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Rope;

    public:
		RopeContext rope_context;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

		OperatorDefinition(const RopeContext &rope_context): 
				OperatorNode(TYPE), rope_context(rope_context) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			spy_assert(num_input() == 2, "Expect the number of operands to be 2 (cur: {})", num_input());

			const Tensor &operand_0 = input(0).tensor;

			const auto &shape_0     = operand_0.get_shape();
			spy_assert(shape_0.dim == 3, 
						"Expect the dimension of operand 0 to be larger than 3 (operand: {})",
						shape_0.dim);
			const Shape shape_res     = shape_0;
			return { shape_res, nullptr };
		}
    };

} // namespace spy