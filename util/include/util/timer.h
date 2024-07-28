#pragma once

#include <cstddef>
#include <chrono>

#ifdef __linux__
	#include <liburing.h>
	#undef BLOCK_SIZE
#endif // __linux__

#include "util/shell/logger.h"

namespace spy {

	struct Timer {
	public:
		using TimePoint = std::chrono::steady_clock::time_point;
		using Duration  = std::chrono::steady_clock::duration;

	public:
		TimePoint start_point;
		Duration  acc_duration;

	public:
		Timer(): acc_duration(0) {}

	public:
		void start() { 
			start_point = std::chrono::steady_clock::now(); 
		}

		void end() { 
			const auto end_point = std::chrono::steady_clock::now(); 
			acc_duration += end_point - start_point;
		}

		template<class T_time>
		size_t get_time_num() const { return std::chrono::duration_cast<T_time>(acc_duration).count(); }
	};

	struct PeriodTimer {
	public:
		std::string name;
		Timer       timer;
	public:
		PeriodTimer(std::string_view name): name(name) { timer.start(); }

		~PeriodTimer() noexcept {
			timer.end();
			spy_info("{} timing: {} ms", name, timer.get_time_num<std::chrono::milliseconds>());
		}
	};

	#define TIMING(name, expression) do { PeriodTimer period_timer(name); expression; } while (0)

	struct PerformanceTimer {
	public:
		Timer model_load_timer;
		Timer sample_timer;
		Timer prefill_timer;
		Timer decode_timer;

		size_t num_sample;
		size_t num_prefill;
		size_t num_decode;

	public:
		PerformanceTimer() : num_sample(-1), num_prefill(-1), num_decode(-1) {}

	public:
		void print_timing() const {
			const size_t model_load_ms = model_load_timer.get_time_num<std::chrono::milliseconds>();
			const size_t sample_ms     = sample_timer.get_time_num<std::chrono::milliseconds>();
			const size_t prefill_ms    = prefill_timer.get_time_num<std::chrono::milliseconds>();
			const size_t decode_ms     = decode_timer.get_time_num<std::chrono::milliseconds>();

			spy_info("[Performance Timer] model loading: {:>16} ms", model_load_ms);
			spy_info("[Performance Timer] sample stage:  {:>16} ms, {} token/s",
				         sample_ms, static_cast<float>(num_sample * 1000) / static_cast<float>(sample_ms));
			spy_info("[Performance Timer] prefill stage: {:>16} ms, {} token/s",
				         prefill_ms, static_cast<float>(num_prefill * 1000) / static_cast<float>(prefill_ms));
			spy_info("[Performance Timer] decode stage:  {:>16} ms, {} token/s",
				         decode_ms, static_cast<float>(num_decode * 1000) / static_cast<float>(decode_ms));			
		}
	};


	
#ifdef __linux__

	template <class Rep, class Period>
	struct __kernel_timespec
	duration_to_kernel_timespec(std::chrono::duration<Rep, Period> dur) {
		struct __kernel_timespec ts;
		auto secs = std::chrono::duration_cast<std::chrono::seconds>(dur);
		auto nsecs =
			std::chrono::duration_cast<std::chrono::nanoseconds>(dur - secs);
		ts.tv_sec = static_cast<std::uint64_t>(secs.count());
		ts.tv_nsec = static_cast<std::uint64_t>(nsecs.count());
		return ts;
	}

	template <class Clk, class Dur>
	struct __kernel_timespec
	timepoint_to_kernel_timespec(std::chrono::time_point<Clk, Dur> tp) {
		return duration_to_kernel_timespec(tp.time_since_epoch());
	}

#endif // __linux__

}  // namespace spy