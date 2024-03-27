#pragma once

#include <cstddef>
#include <chrono>

#include "util/logger.h"

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
			SPY_INFO_FMT("{} timing: {} ms", name, timer.get_time_num<std::chrono::milliseconds>());
		}
	};

	#define TIMING(name, expression) do { PeriodTimer period_timer(name); expression; } while (0)

	struct PerformanceTimer {
	public:
		Timer model_load_timer;
		Timer sample_timer;
		Timer prefill_timer;
		Timer decode_timer;

		size_t model_bytes;
		size_t num_sample;
		size_t num_prefill;
		size_t num_decode;

	public:
		PerformanceTimer() : model_bytes(-1), num_sample(-1), num_prefill(-1), num_decode(-1) {}

	public:
		void print_timing() const {
			const size_t model_load_ms = model_load_timer.get_time_num<std::chrono::milliseconds>();
			const size_t sample_ms     = sample_timer.get_time_num<std::chrono::milliseconds>();
			const size_t prefill_ms    = prefill_timer.get_time_num<std::chrono::milliseconds>();
			const size_t decode_ms     = decode_timer.get_time_num<std::chrono::milliseconds>();

			SPY_INFO_FMT("[Performance Timer] model loading:\t {} ms, {} MB/s",
				         model_load_ms, static_cast<float>(model_bytes) / static_cast<float>(model_load_ms) / 1e3);
			SPY_INFO_FMT("[Performance Timer] sample stage:\t {} ms, {} token/s",
				         sample_ms, static_cast<float>(num_sample * 1000) / static_cast<float>(sample_ms));
			SPY_INFO_FMT("[Performance Timer] prefill stage:\t {} ms, {} token/s",
				         prefill_ms, static_cast<float>(num_prefill * 1000) / static_cast<float>(prefill_ms));
			SPY_INFO_FMT("[Performance Timer] decode stage:\t {} ms, {} token/s",
				         decode_ms, static_cast<float>(num_decode * 1000) / static_cast<float>(decode_ms));			
		}
	};
}  // namespace spy