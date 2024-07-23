#pragma once

#include <cstddef>

namespace spy {

	inline static constexpr unsigned long long operator""_KB(unsigned long long val) { return val * 1000; }
	inline static constexpr unsigned long long operator""_MB(unsigned long long val) { return val * 1000'000; }
	inline static constexpr unsigned long long operator""_GB(unsigned long long val) { return val * 1000'000'000; }

	inline static constexpr float operator""_KBf(unsigned long long val) { return static_cast<float>(val) * 1000.f; }
	inline static constexpr float operator""_MBf(unsigned long long val) { return static_cast<float>(val) * 1000'000.f; }
	inline static constexpr float operator""_GBf(unsigned long long val) { return static_cast<float>(val) * 1000'000'000.f; }

}  // namespace spy