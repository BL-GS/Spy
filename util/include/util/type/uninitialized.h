/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <utility>

#include "util/type/non_void.h"

namespace spy {

	template <class T>
	struct Uninitialized {
	public:
		/// Wrap value with union in case of automatic initialization
		union { T value; };

	public:
		Uninitialized() noexcept {}

		Uninitialized(Uninitialized &&) = delete;

		~Uninitialized() noexcept = default;

	public:
		T move_value() {
			T ret(std::move(value));
			value.~T();
			return ret;
		}

		template <class... Args>
		void put_value(Args &&...args) {
			new (std::addressof(value)) T(std::forward<Args>(args)...);
		}
	};

	template <>
	struct Uninitialized<void> {
	public:
		auto move_value() {
			return NonVoidHelper<>{};
		}

		void put_value(NonVoidHelper<>) {}
	};

	template <class T>
	struct Uninitialized<T const> : Uninitialized<T> {};

	template <class T>
	struct Uninitialized<T &> : Uninitialized<std::reference_wrapper<T>> {};

	template <class T>
	struct Uninitialized<T &&> : Uninitialized<T> {};

} // namespace spy