#pragma once

#include "util/logger.h"
#include "operator/type.h"
#include "operator/config.h"
#include "operator/operator.h"
#include "graph/graph.h"

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Nop> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Nop;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit, std::string_view name): OperatorNode(credit, name, TYPE) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			SPY_ASSERT_FMT(input.size() == 1, "Expect the number of operands to be 1 (cur: {})", input.size());

			const Tensor &operand = get_tensor_from_node(input[0]);
			return { operand.get_shape(), nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::GetRow> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::GetRow;

	public:
		NumberType type_res = NumberType::FP32;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit, std::string_view name, NumberType type_res = NumberType::FP32): 
			OperatorNode(credit, name, TYPE), type_res(type_res) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			SPY_ASSERT_FMT(input.size() == 2, "Expect the number of operands to be 2 (cur: {})", input.size());

			const Tensor &operand_0 = get_tensor_from_node(input[0]);
			const Tensor &operand_1 = get_tensor_from_node(input[1]);

			const auto &shape_0 = operand_0.get_shape();
			const auto &shape_1 = operand_1.get_shape();
			const auto  type_1  = operand_1.get_number_type();
			
			SPY_ASSERT_FMT(type_1 == NumberType::INT32, 
					"Expect the index to be of type INT32 (cur: {})", 
					magic_enum::enum_name(type_1));

			const auto num_element = Shape::DimensionArray {
				shape_0.elements[0], shape_1.elements[0], shape_1.elements[1], shape_1.elements[2]
			};

			const Shape shape_res(shape_1.dim + 1, num_element, type_res);
			return { shape_res, nullptr };
		}        
    };


    template<>
    struct OperatorDefinition<OperatorType::Dup> final: OperatorNode {
	public:
		static constexpr OperatorType TYPE = OperatorType::Dup;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit, std::string_view name): OperatorNode(credit, name, TYPE) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			SPY_ASSERT_FMT(input.size() == 1, "Expect the number of operands to be 1 (cur: {})", input.size());	
			const Tensor &operand_0 = get_tensor_from_node(input[0]);
			const auto &shape_0     = operand_0.get_shape();

			const Shape shape_res     = shape_0;
			return { shape_res, nullptr };
		}
    };

    template<>
    struct OperatorDefinition<OperatorType::Copy> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Copy;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit, std::string_view name): OperatorNode(credit, name, TYPE) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			SPY_ASSERT_FMT(input.size() == 2, "Expect the number of operands to be 2 (cur: {})", input.size());	

			const Tensor &operand_0 = get_tensor_from_node(input[0]);
			const Tensor &operand_1 = get_tensor_from_node(input[1]);

			const Shape &shape_0 = operand_0.get_shape();
			const Shape &shape_1 = operand_1.get_shape();

			SPY_ASSERT_FMT(shape_0.total_element() == shape_1.total_element(), 
				"Expect the size of operands to be the same (operand_0: {}, operand_1: {})",
				shape_0.to_string(), shape_1.to_string()
			);
			// No output
			return { shape_1, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::Reshape> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Reshape;

	public:
		Shape new_shape;

    public:
		OperatorDefinition() = default;

        template<class ...Args>
		OperatorDefinition(NodeCredit credit, std::string_view name, Args &&...args): 
				OperatorNode(credit, name, TYPE), new_shape(std::forward<Args>(args)...) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            SPY_ASSERT_FMT(input.size() == 1, "Expect the number of operands to be 1 (cur: {})", input.size());

			const Tensor &operand_0 = get_tensor_from_node(input[0]);

            const Shape &shape_0  = operand_0.get_shape();
            const Shape shape_res = new_shape;

			SPY_ASSERT_FMT(shape_res.total_element() == shape_0.total_element(),
			           "Result and operands should be of the same number of elements (operand: {}, result: {})",
			           shape_res.to_string(), shape_0.to_string());

			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::View> final: OperatorNode {
    public:
        static constexpr int64_t	INVALID_OFFSET 	= std::numeric_limits<int64_t>::max();
		static constexpr OperatorType TYPE          = OperatorType::View;

	public:
        int64_t offset = INVALID_OFFSET;
		Shape new_shape;

	public:
		OperatorDefinition() = default;

		template<class ...Args>
		OperatorDefinition(NodeCredit credit, std::string_view name, int64_t offset, Args &&...args):
				OperatorNode(credit, name, TYPE), offset(offset), new_shape(std::forward<Args>(args)...) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            SPY_ASSERT_FMT(input.size() == 1, "Expect the number of operands to be 1 (cur: {})", input.size());

            const Tensor &operand_0 = get_tensor_from_node(input[0]);
            const Shape &shape_0    = operand_0.get_shape();
            const Shape shape_res   = new_shape;

			SPY_ASSERT_FMT(shape_res.total_size() + offset <= shape_0.total_size(),
			           "Operand should be within the range of the result (operand: {}, result: {}, offset: {})",
			           shape_0.to_string(), shape_res.to_string(), offset);

			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::Transpose> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Transpose;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit, std::string_view name): OperatorNode(credit, name, TYPE) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			SPY_ASSERT_FMT(input.size() == 1, "Expect the number of operands to be 1 (cur: {})", input.size());	
			const Tensor &operand_0 = get_tensor_from_node(input[0]);
			const auto &shape_0     = operand_0.get_shape();

			Shape shape_res = shape_0;
			shape_res.transpose();
			return { shape_res, nullptr };
		}
    };

    
    template<>
    struct OperatorDefinition<OperatorType::Permute> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Permute;

	public:
		std::array<size_t, MAX_DIM> axis;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit, std::string_view name, const std::array<size_t, MAX_DIM> &axis): 
				OperatorNode(credit, name, TYPE), axis(axis) {}

		template<class T>
		OperatorDefinition(NodeCredit credit, std::string_view name, const std::initializer_list<T> &new_axis): 
				OperatorNode(credit, name, TYPE), axis{0} {
			SPY_ASSERT_FMT(new_axis.size() == MAX_DIM, "Expect the initializer list to be of dim {} (cur: {})", MAX_DIM, new_axis.size());
			auto iter_new = new_axis.begin();
			for (size_t i = 0; i < MAX_DIM; ++i) {
				axis[i] = *iter_new;
				++iter_new;
			}
		}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			SPY_ASSERT_FMT(input.size() == 1, "Expect the number of operands to be 1 (cur: {})", input.size());	
			const Tensor &operand_0 = get_tensor_from_node(input[0]);
			const auto &shape_0     = operand_0.get_shape();

			Shape shape_res = shape_0;
			shape_res.permute(axis);
			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::Contiguous> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Contiguous;

	public:
		Shape new_shape;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit, std::string_view name, const Shape &new_shape): 
				OperatorNode(credit, name, TYPE), new_shape(new_shape) {}

		template<class ...Args>
		OperatorDefinition(NodeCredit credit, std::string_view name, Args &&...args): 
				OperatorNode(credit, name, TYPE), new_shape(std::forward<Args>(args)...) { }

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            SPY_ASSERT_FMT(input.size() == 1, "Expect the number of operands to be 1 (cur: {})", input.size());

			const Tensor &operand_0 = get_tensor_from_node(input[0]);

            const Shape &shape_0  = operand_0.get_shape();
            const Shape shape_res = new_shape;

			SPY_ASSERT_FMT(shape_0.total_element() == shape_res.total_element(), 
				"Expect the total number of element of operand and result to be the same. (operand: {}, result: {})",
				shape_0.to_string(), shape_res.to_string()
			);
			
			return { shape_res, nullptr };
		}
    };

} // namespace spy