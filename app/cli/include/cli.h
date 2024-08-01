#pragma once

#include <memory>

#include "util/timer.h"
#include "loader/loader.h"
#include "llm/sampler/sampler.h"
#include "llm/model/model.h"
#include "distributor/distributor.h"
#include "backend/config.h"
#include "perf/perfetto/trace.h"

namespace spy {

    class AutoModelGenerator {
    public:
        /* Model Loader */
        std::unique_ptr<ModelLoader>    loader_ptr;
        std::unique_ptr<AbstractModel>  model_ptr;
        std::unique_ptr<Sampler>        sampler_ptr;
        /* Graph Constructor and Executor */
        GraphStorage graph_storage;
        Graph        graph;
        std::unique_ptr<AbstractGraphDistributor> distributor_ptr;
        /* Profiler */
        PerformanceTimer perf_timer;

    public:
        AutoModelGenerator(std::string_view filename, const HyperParam &param): graph(0, graph_storage) {
            loader_ptr  = ModelLoaderFactory::build_model_loader("simple", filename);
            model_ptr   = ModelBuilder::build_model(loader_ptr->context, param);
            sampler_ptr = SamplerFactory::build_sampler(SamplerType::Greedy);

            perf_timer.num_sample  = 0;
            perf_timer.num_prefill = 0;
            perf_timer.num_decode  = 0;

            distributor_ptr = GraphDistributorFactory::build_graph_distributor("simple", loader_ptr.get());

            loader_ptr->preload();
        }

        ~AutoModelGenerator() noexcept = default;

    public:
        void add_backend(AbstractBackend *backend_ptr, std::string_view policy) {
            distributor_ptr->add_backend(backend_ptr, policy);
        }

        template<class T_Stream>
        void generate(const std::string &prompt, size_t max_num_predict, T_Stream &stream) {
            spy_start_tracing();
            spy_enable_tracing();

            const auto &model_metadata = model_ptr->get_info();

            ModelIO model_io;
            auto token_id_array = tokenize(prompt, true, false);
            for (size_t i = 0; i < token_id_array.size(); ++i) {
                model_io.add(token_id_array[i], i, { 0 }, false);
            }
            model_io.enable_logits.back() = true;
            perf_timer.num_prefill = token_id_array.size();

            model_ptr->build_graph(loader_ptr->context, graph, model_io);

            distributor_ptr->prepare_graph(std::addressof(graph));

            stream << prompt << std::flush;

            int32_t cur_pos = 0;
            for (uint32_t predict_idx = 0; predict_idx < max_num_predict; ++predict_idx) {
                const size_t num_token = model_io.num_token();
                const size_t num_vocab = model_metadata.num_vocab;

                /* Predict */
                auto &predict_timer = (predict_idx == 0) ? perf_timer.prefill_timer : perf_timer.decode_timer;
                predict_timer.start();

                distributor_ptr->execute();

                if (predict_idx != 0) { ++perf_timer.num_decode; }
                predict_timer.end();

                /* Get logits of the last token */
                spy_assert(model_io.logits.size() >= num_vocab);
                const std::span<const float> logits{model_io.logits.cend() - num_vocab, model_io.logits.cend()};

                /* Sample */
                perf_timer.sample_timer.start();
                const TokenID new_token_id = sample(logits);
                perf_timer.num_sample++;
                perf_timer.sample_timer.end();


                /* Output predicted token */
                stream << token_to_word(new_token_id) << std::flush;

                /* Predict */
                model_io.reset();
                cur_pos += num_token;
                model_io.add(new_token_id, cur_pos, { 0 }, true);

                model_ptr->propagate(graph, model_io);
            }

            spy_stop_tracing("spy.perfetto-trace");
        }

    private:
        void load() {
            perf_timer.model_load_timer.start();
            loader_ptr->preload();
            perf_timer.model_load_timer.end();
        }

        std::vector<TokenID> tokenize(const std::string &text, bool add_bos, bool special) {
            auto &tokenizer = model_ptr->get_tokenizer();
            return tokenizer.tokenize(text, add_bos, special);
        }

        std::string token_to_word(TokenID token_id) const {
            auto &tokenizer = model_ptr->get_tokenizer();
            if (token_id == tokenizer.get_special_eos_id()) { return "\n"; }
            return tokenizer.token_to_piece(token_id);
        }

        TokenID sample(std::span<const float> logits) {
            const size_t num_vocab = logits.size();
            auto &tokenizer = model_ptr->get_tokenizer();

            TokenCandidateArray candidates(num_vocab);
            for (size_t token_id = 0; token_id < num_vocab; ++token_id) {
                candidates[token_id] = {
                    .token_id = static_cast<TokenID>(token_id),
                    .logit    = logits[token_id]
                };
            }
            TokenID new_token_id = sampler_ptr->sample(candidates);

            spy_debug(DebugFlag::Execute, "Sample out: token_id {}, logit {}, token {}", new_token_id, candidates[new_token_id].logit, tokenizer.token_to_piece(new_token_id));
            return new_token_id;
        }

    };

} // namespace spy