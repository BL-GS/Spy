#pragma once

#include "operator/type.h"
#include "operator/config.h"
#include "graph/op_node.h"
#include "operator/common.h"

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Add> final: OperatorBinaryNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Add;

	public:
	    OperatorDefinition(): OperatorBinaryNode(TYPE) {}

		~OperatorDefinition() noexcept = default;
    };


    template<>
    struct OperatorDefinition<OperatorType::Sub> final: OperatorBinaryNode {
	public:
		static constexpr OperatorType TYPE = OperatorType::Sub;

	public:
	    OperatorDefinition(): OperatorBinaryNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;
    };


    template<>
    struct OperatorDefinition<OperatorType::Mul> final: OperatorBinaryNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Mul;

	public:
	    OperatorDefinition(): OperatorBinaryNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;
    };


    template<>
    struct OperatorDefinition<OperatorType::Div> final: OperatorBinaryNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Div;

	public:
	    OperatorDefinition(): OperatorBinaryNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;
    };

} // namespace spy