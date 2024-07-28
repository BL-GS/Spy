#include <cstdint>
#include <iostream>
#include <string>

#include "util/shell/cmdline.h" 
#include "util/shell/logger.h"
#include "backend/config.h"
#include "cli.h"

using namespace spy;

int main(int argc, char **argv) {
	/* Parsing arguments */
	Argument cmdline_argument;
	cmdline_argument.parse_argv(argc, argv);

	/* Logger */
	init_logger_format(cmdline_argument.get_arg<int>("--log-level"));

	/* Initialize backend */
	BackendFactory backend_factory;
	ConfigTable cpu_backend_config;
	const uint32_t num_thread = cmdline_argument.get_arg<uint32_t>("--num-thread");
	cpu_backend_config.add("num_thread", std::to_string(num_thread));
	auto cpu_backend_ptr = backend_factory.init_backend("cpu:default", cpu_backend_config);

#ifdef SPY_BACKEND_CUDA
	ConfigTable gpu_backend_config;
	auto gpu_backend_ptr = backend_factory.init_backend("gpu:default", gpu_backend_config);
#endif

	// The hyper parameters defined by user will overwrite that read from model file.
	HyperParam hyper_param {
		.num_context			= cmdline_argument.get_arg<uint32_t>("--num-context"),

		.rope_scaling_type      = HyperParam::parse_rope_scaling_type(cmdline_argument.get_arg<std::string>("--rope-scaling-type")),
        .rope_pooling_type      = HyperParam::parse_pooling_type(cmdline_argument.get_arg<std::string>("--rope-pooling-type")),
		.rope_freq_base 		= cmdline_argument.get_arg<float>("--rope-freq-base"),
		.rope_freq_scale 		= cmdline_argument.get_arg<float>("--rope-freq-scale"),
		.yarn_ext_factor		= cmdline_argument.get_arg<float>("--yarn-ext-factor"),
		.yarn_attn_factor		= cmdline_argument.get_arg<float>("--yarn-attn-factor"),
		.yarn_beta_fast			= cmdline_argument.get_arg<float>("--yarn-beta-fast"),
		.yarn_beta_slow			= cmdline_argument.get_arg<float>("--yarn-beta-slow"),
		.yarn_orig_ctx			= cmdline_argument.get_arg<uint32_t>("--yarn-orig-ctx")
	};

	/* Load model */
	const auto model_filename = cmdline_argument.get_arg<std::string>("--model");

	/* Initialize input */
	const auto prompt = cmdline_argument.get_arg<std::string>("--prompt");

	/* Decode stage */
	const uint32_t num_predict = cmdline_argument.get_arg<uint32_t>("--num-predict");

	AutoModelGenerator model(model_filename, hyper_param);
	model.generate(prompt, num_predict, std::cout);

	/* Finish */
	std::cout << std::endl;

	model.perf_timer.print_timing();
	spy_info("Finished prediction");
	spy_info("Exit...");

	return 0;
}
