#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <span>

#include "util/cmdline.h" 
#include "util/logger.h"
#include "util/timer.h"
#include "backend/cpu/cpu.h"
#include "graph/scheduler.h"
#include "model/file/loader.h"
#include "model/file/mapper.h"
#include "model/sample/config.h"
#include "model/model_impl/model.h"
#include "model/model_impl/config.h"
#include "model/sample/sampler.h"


using namespace spy;

auto make_model_from_file(const std::string &model_filename, const HyperParam &hyper_param) {
	/* Load model */
	auto [context_ptr, model_type] = GGUFLoader::init_from_file(model_filename);
	/* Build model */
	auto model_ptr = ModelBuilder::build_model(model_type, std::move(context_ptr), hyper_param);
	return model_ptr;
}

void decode(std::unique_ptr<AbstractModel> &model_ptr, ModelIO &model_io, CPUBackend *cpu_backend_ptr) {
	auto graph_ptr = model_ptr->build_graph(model_io);
	/* Build up a scheduler for graph execution */
	DefaultGraphScheduler schedule({cpu_backend_ptr});
	/* Allocate space for tensors */
	schedule.reserve(graph_ptr.get());
	/* Execute operations of the graph */
	schedule.execute(graph_ptr.get());
}

int main(int argc, char **argv) {
	/* Parsing arguments */
	Argument cmdline_argument;
	cmdline_argument.parse_argv(argc, argv);

	/* Timer setup */
	PerformanceTimer perf_timer;
	perf_timer.model_bytes = 0;
	perf_timer.num_sample  = 0;
	perf_timer.num_prefill = 0;
	perf_timer.num_decode  = 0;

	/* Initialize backend */
	const uint32_t num_thread = cmdline_argument.get_arg<uint32_t>("--num-thread");
	std::unique_ptr<CPUBackend> cpu_backend_ptr = std::make_unique<DefaultCPUBackend>(num_thread, 0);

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
	perf_timer.model_load_timer.start();

	const auto model_filename = cmdline_argument.get_arg<std::string>("--model");
	auto model_ptr   = make_model_from_file(model_filename, hyper_param);
	auto &tokenizer  = model_ptr->get_tokenizer();
	auto sampler_ptr = SamplerFactory::build_sampler(SamplerType::Greedy);
	const auto &model_metadata = model_ptr->get_info();
	// Load data
	ModelMapper model_mapper(model_filename);
	model_mapper.mapping(model_ptr->get_context());

	perf_timer.model_bytes = model_mapper.file.size();
	perf_timer.model_load_timer.end();

	/* Initialize input */
	const auto prompt = cmdline_argument.get_arg<std::string>("--prompt");
	ModelIO model_io;
	auto token_id_array = tokenizer.tokenize(prompt, true, false);
	for (size_t i = 0; i < token_id_array.size(); ++i) {
		model_io.add(token_id_array[i], i, { 0 }, false);
	}
	model_io.enable_logits.back() = true;

	/* Prefill stage */
	std::cout << prompt << std::flush;
	perf_timer.num_prefill = prompt.size();

	/* Decode stage */
	const uint32_t num_predict = cmdline_argument.get_arg<uint32_t>("--num-predict");
	int32_t cur_pos = 0;
	for (uint32_t predict_idx = 0; predict_idx < num_predict; ++predict_idx) {
		const size_t num_token = model_io.num_token();
		const size_t num_vocab = model_metadata.num_vocab;

		/* Predict */
		auto &predict_timer = (predict_idx == 0) ? perf_timer.prefill_timer : perf_timer.decode_timer;
		predict_timer.start();

		decode(model_ptr, model_io, std::to_address(cpu_backend_ptr));

		if (predict_idx != 0) { ++perf_timer.num_decode; }
		predict_timer.end();

		/* Get logits of the last token */
		const std::span<const float> logits{model_io.logits.cend() - num_vocab, model_io.logits.cend()};

		/* Sample */
		auto &sample_timer = perf_timer.sample_timer;
		sample_timer.start();

		TokenCandidateArray candidates(num_vocab);
		for (size_t token_id = 0; token_id < num_vocab; ++token_id) {
			candidates[token_id] = {
				.token_id = static_cast<TokenID>(token_id),
				.logit    = logits[token_id]
			};
		}
		const TokenID new_token_id = sampler_ptr->sample(candidates);			

		++perf_timer.num_sample;
		sample_timer.end();

		SPY_DEBUG_FMT_OPTION(Execute, "Sample out: token_id {}, logit {}, token {}", new_token_id, candidates[new_token_id].logit, tokenizer.token_to_piece(new_token_id));

		/* Output predicted token */
		if (new_token_id == tokenizer.get_special_eos_id()) { std::cout << std::endl; break; }
		std::cout << tokenizer.token_to_piece(new_token_id) << std::flush;

		/* Predict */
		model_io.reset();
		cur_pos += num_token;
		model_io.add(new_token_id, cur_pos, { 0 }, true);
	}

	/* Finish */
	std::cout << std::endl;

	perf_timer.print_timing();
	SPY_INFO("Finished prediction");
	SPY_INFO("Exit...");

	return 0;
}
