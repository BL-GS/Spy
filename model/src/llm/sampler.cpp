#include <magic_enum_switch.hpp>

#include "llm/sampler/abstract_sample.h"
#include "llm/sampler/greedy.h"
#include "llm/sampler/sampler.h"

namespace spy {

    std::unique_ptr<Sampler> SamplerFactory::build_sampler(SamplerType sampler_type) {
        return magic_enum::enum_switch([](const auto T_sampler_type){
            return static_cast<std::unique_ptr<Sampler>>(std::make_unique<SamplerImpl<T_sampler_type>>());
        }, sampler_type);
    }

} // namespace spy