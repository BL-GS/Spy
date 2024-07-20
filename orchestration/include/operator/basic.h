#pragma once

#include "operator/type.h"
#include "operator/config.h"
#include "operator/common.h"

#ifndef OPERATOR_HEADER_MACRO
	#warning "Do not include basic.h manually, please use operator/operator.h instead."
#endif // OPERATOR_HEADER_MACRO

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Add> final: OperatorBinaryNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Add;

	public:
	    OperatorDefinition(): OperatorBinaryNode(TYPE) {}

		~OperatorDefinition() noexcept = default;
    };
	using AddOpDef = OperatorDefinition<OperatorType::Add>;


    template<>
    struct OperatorDefinition<OperatorType::Sub> final: OperatorBinaryNode {
	public:
		static constexpr OperatorType TYPE = OperatorType::Sub;

	public:
	    OperatorDefinition(): OperatorBinaryNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;
    };
	using SubOpDef = OperatorDefinition<OperatorType::Sub>;


    template<>
    struct OperatorDefinition<OperatorType::Mul> final: OperatorBinaryNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Mul;

	public:
	    OperatorDefinition(): OperatorBinaryNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;
    };
	using MulOpDef = OperatorDefinition<OperatorType::Mul>;


    template<>
    struct OperatorDefinition<OperatorType::Div> final: OperatorBinaryNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Div;

	public:
	    OperatorDefinition(): OperatorBinaryNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;
    };
	using DivOpDef = OperatorDefinition<OperatorType::Div>;

} // namespace spy