/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

namespace spy {

	template <class... Ts>
	struct TypeList {};

	template <class Last>
	struct TypeList<Last> {
		using FirstType = Last;
		using LastType  = Last;
	};

	template <class First, class... Ts>
	struct TypeList<First, Ts...> {
		using FirstType = First;
		using LastType  = typename TypeList<Ts...>::LastType;
	};

} // namespace spy