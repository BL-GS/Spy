/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstdint>
#include <spdlog/common.h>
#include <string_view>
#include <stdexcept>
#include <argparse/argparse.hpp>
#include <sys/types.h>

#include "util/shell/logger.h"

namespace spy {

	class Argument {

	private:
		argparse::ArgumentParser parser_;

	public:
		Argument(std::string_view name = "Spy") : parser_(name.data()) { 
			spy_info("Initialize command line arguments parser");
			define_augment();
		}

	public: /* Augment definition */
		void define_augment() {
			parser_.add_argument("--log-level").default_value(2).scan<'i', int32_t>()
				.help("The lower bound of log to output(1: debug; 2: info; 3: warn; 4: error; 5: fatal; 6: off)")
				.choices(1, 2, 3, 4, 5, 6);

			parser_.add_argument("-m", "--model").required()
				.help("The path of the model file");
			parser_.add_argument("-p", "--prompt").default_value("Once upon a time")
				.help("The prompt as the input to the model");
			parser_.add_argument("-n", "--num-predict").default_value(1U).scan<'u', uint32_t>()
				.help("The number of token to be predicted");

			parser_.add_argument("-c", "--num-context").default_value(0U).scan<'u', uint32_t>()
				.help("The maximum context of generative LLM");

			parser_.add_argument("--rope-scaling-type").default_value("unspecified")
				.help("The type of rope scaling");
			parser_.add_argument("--rope-pooling-type").default_value("unspecified")
				.help("The pooling type of embeddings");
			parser_.add_argument("--rope-freq-base").default_value(0.0F).scan<'f', float>()
				.help("The YoRN base frequency");
			parser_.add_argument("--rope-freq-scale").default_value(0.0F).scan<'f', float>()
				.help("The YoRN frequency scaling factor");
			parser_.add_argument("--yarn-ext-factor").default_value(-1.0F).scan<'f', float>()
				.help("The YoRN extrapolation mix factor");
			parser_.add_argument("--yarn-attn-factor").default_value(1.0F).scan<'f', float>()
				.help("The YoRN magnitude scaling factor");
			parser_.add_argument("--yarn-beta-fast").default_value(32.0F).scan<'f', float>()
				.help("The YoRN low correction dim");
			parser_.add_argument("--yarn-beta-slow").default_value(1.0F).scan<'f', float>()
				.help("The YoRN high correction dim");
			parser_.add_argument("--yarn-orig-ctx").default_value(0U).scan<'u', uint32_t>()
				.help("The YoRN original context length");

			parser_.add_argument("-t", "--num-thread").required().scan<'u', uint32_t>()
				.help("The number of thread");
		}

	public:
		void parse_argv(int argc, char **argv) {
			spy_info("Parse command line arguments");
			try {
				parser_.parse_args(argc, argv);
			} catch (const std::exception &err) {
				std::cerr << "Args parse error: " << err.what() << std::endl;
				std::cerr << parser_ << std::endl;
				throw SpyException("Argparse Fault");
			}
		}

		template<class T>
		T get_arg(std::string_view key) const { return parser_.get<T>(key); }

	};
	
}  // namespace spy