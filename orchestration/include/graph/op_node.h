#pragma once

#include "operator/type.h"
#include "graph/basic_node.h"

namespace spy {

	struct OperatorNode: BasicNode {
	public: /* Content */
		/// The type of operation
		OperatorType 	op_type;

	public:
		OperatorNode() : op_type(OperatorType::Nop) {}

		OperatorNode(OperatorType op_type): op_type(op_type) {}

		OperatorNode(const OperatorNode &other) = default;

		virtual ~OperatorNode() noexcept = default;

	public:
		std::map<std::string_view, std::string> property() const override;
	};

} // namespace spy