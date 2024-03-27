/*
 * @author: BL-GS 
 * @date:   2024/3/31
 */

#pragma once

namespace spy {

	template<class T>
	inline constexpr T align_floor(T val, T align) { return val / align * align; }

	template<class T>
	inline constexpr T align_ceil(T val, T align) { return (val + align - 1) / align * align; }

	template<class T>
	inline constexpr T div_floor(T val, T div) { return val / div; }

	template<class T>
	inline constexpr T div_ceil(T val, T div) { return (val + div) / div; }

	template<class T>
	inline constexpr T div_floor_pow2(T val, T bit) { return val >> bit; }

}  // namespace spy
