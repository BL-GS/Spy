#include <cstdint>
#include <iostream>
#include <string>

#include "util/log/logger.h"
#include "backend/backend.h"
#include "cli.h"
#include "cmdline.h"

using namespace spy;

int main(int argc, char **argv) {
	/* Parsing arguments */
	Argument cmdline_argument;
	cmdline_argument.parse_argv(argc, argv);

	/* Logger */
	init_logger_format(cmdline_argument.get_log_param());

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

	/* Load model */
	const auto model_filename = cmdline_argument.get_arg<std::string>("--model");

	const PredictParam predict_param = cmdline_argument.get_predict_param();

	AutoModelGenerator model(model_filename, cmdline_argument.get_hyper_param());
	model.add_backend(cpu_backend_ptr.get(), "greedy");
	model.generate(predict_param.prompt, predict_param.num_predict, std::cout);

	/* Finish */
	std::cout << std::endl;

	model.perf_timer.print_timing();
	spy_info("Finished prediction");
	spy_info("Exit...");

	return 0;
}
