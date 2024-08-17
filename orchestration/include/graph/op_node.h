#pragma once

#include "util/log/logger.h"
#include "operator/type.h"
#include "graph/basic_node.h"
#include "graph/data_node.h"
#include <cstddef>

namespace spy {

	struct OperatorNode: BasicNode {
	public: /* Content */
		/// The type of operation
		OperatorType 	op_type;

	public:
		OperatorNode() : op_type(OperatorType::Nop) {}

		OperatorNode(OperatorType op_type): op_type(op_type) {}

		OperatorNode(const OperatorNode &other) = default;

		~OperatorNode() noexcept override = default;

	public: /* Assertion */
		template<class ...Args>
		void assert_num_input(size_t expect, Args ...others) const {
			const size_t cur = num_input();
			if constexpr (sizeof...(others) == 0) {
				spy_assert(cur == expect, "invalid number of inputs {} (expect: {})", cur, expect);
			} else {
				if (cur != expect) { assert_num_input(others...); }
			}
		}

		template<class ...Args>
		void assert_num_output(size_t expect, Args ...others) const {
			const size_t cur = num_output();
			if constexpr (sizeof...(others) == 0) {
				spy_assert(cur == expect, "invalid number of outputs {} (expect: {})", cur, expect);
			} else {
				if (cur != expect) { assert_num_output(others...); }
			}
		}

	public:
		DataNode *input_data(size_t idx) const {
			return input<DataNode>(idx);
		}

		DataNode *output_data(size_t idx) const {
			return output<DataNode>(idx);
		}

	public:
		virtual void propagate() = 0;

	public:
		std::map<std::string_view, std::string> property() const override;
	};

} // namespace spy