#pragma once

#include <argparse/argparse.hpp>

#include "util/log/logger.h"
#include "llm/model/config.h"

namespace spy {

    struct PredictParam {
        std::string prompt;
        uint32_t    num_predict;
        uint32_t    num_context;
    };

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
            define_predict_augment();
            define_model_augment();
            define_backend_augment();
            define_log_augment();
        }

        void define_predict_augment() {
            parser_.add_argument("-p", "--prompt")
                .default_value("Once upon a time")
                .help("The prompt as the input to the model");
            parser_.add_argument("-n", "--num-predict")
                .default_value(1U).scan<'u', uint32_t>()
                .help("The number of token to be predicted");
            parser_.add_argument("-c", "--num-context")
                .default_value(512U).scan<'u', uint32_t>()
                .help("The maximum context of generative LLM");
        }

        void define_backend_augment() {
            parser_.add_argument("-t", "--num-thread")
                .required()
                .scan<'u', uint32_t>()
                .help("The number of thread");
        }

        void define_model_augment() {
            parser_.add_argument("-m", "--model")
                .required()
                .help("The path of the model file");

            parser_.add_argument("--rope-scaling-type")
                .default_value("unspecified")
                .help("The type of rope scaling");
            parser_.add_argument("--rope-pooling-type")
                .default_value("unspecified")
                .help("The pooling type of embeddings");
            parser_.add_argument("--rope-freq-base")
                .default_value(0.0F).scan<'f', float>()
                .help("The YoRN base frequency");
            parser_.add_argument("--rope-freq-scale")
                .default_value(0.0F).scan<'f', float>()
                .help("The YoRN frequency scaling factor");
            parser_.add_argument("--yarn-ext-factor")
                .default_value(-1.0F).scan<'f', float>()
                .help("The YoRN extrapolation mix factor");
            parser_.add_argument("--yarn-attn-factor")
                .default_value(1.0F).scan<'f', float>()
                .help("The YoRN magnitude scaling factor");
            parser_.add_argument("--yarn-beta-fast")
                .default_value(32.0F).scan<'f', float>()
                .help("The YoRN low correction dim");
            parser_.add_argument("--yarn-beta-slow")
                .default_value(1.0F).scan<'f', float>()
                .help("The YoRN high correction dim");
            parser_.add_argument("--yarn-orig-ctx")
                .default_value(0U).scan<'u', uint32_t>()
                .help("The YoRN original context length");
        }

        void define_log_augment() {
            parser_.add_argument("--log-level")
                .default_value("info")
                .help("The lower bound of log to output(debug, info, warn, error, fatal, off)");
        }

	public: /* Augment parsing */
		void parse_argv(int argc, char **argv) { 
			try {
            	parser_.parse_args(argc, argv);
			} catch (const std::exception &err) {
				spy_error("Args parse error: {}", err.what());
				std::cerr << parser_ << std::endl;
				throw SpyException("Argparse Fault");
			}
        }

		template<class T>
		T get_arg(std::string_view key) const { 
            return parser_.get<T>(key); 
        }

        PredictParam get_predict_param() const {
            PredictParam predict_param {
                .prompt = get_arg<std::string>("--prompt"),
                .num_predict = get_arg<uint32_t>("--num-predict"),
                .num_context = get_arg<uint32_t>("--num-context")
            };
            return predict_param;
        }

        LogParam get_log_param() const {
            LogParam log_param {
                .log_level = get_arg<std::string>("--log-level")
            };
            return log_param;
        }

        HyperParam get_hyper_param() const {
            HyperParam hyper_param {
                .num_context			= get_arg<uint32_t>("--num-context"),

                .rope_type              = RopeType::None,
                .rope_scaling_type      = HyperParam::parse_rope_scaling_type(get_arg<std::string>("--rope-scaling-type")),
                .rope_pooling_type      = HyperParam::parse_pooling_type(get_arg<std::string>("--rope-pooling-type")),
                .rope_freq_base 		= get_arg<float>("--rope-freq-base"),
                .rope_freq_scale 		= get_arg<float>("--rope-freq-scale"),
                .yarn_ext_factor		= get_arg<float>("--yarn-ext-factor"),
                .yarn_attn_factor		= get_arg<float>("--yarn-attn-factor"),
                .yarn_beta_fast			= get_arg<float>("--yarn-beta-fast"),
                .yarn_beta_slow			= get_arg<float>("--yarn-beta-slow"),
                .yarn_orig_ctx			= get_arg<uint32_t>("--yarn-orig-ctx"),
            };
            return hyper_param;
        }


	};


} // namespace