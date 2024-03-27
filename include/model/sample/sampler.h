#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "model/sample/type.h"
#include "model/sample/abstract_sample.h"
#include "model/sample/greedy.h"

namespace spy {

    class SamplerFactory {

    public:
        static std::unique_ptr<AbstractSampler> build_sampler(SamplerType sampler_type) {
            switch (sampler_type) {
            case SamplerType::Greedy:
                return std::make_unique<GreedySampler>();
                
            default:
                SPY_WARN_FMT("Unknown sampler type: {}. Use greedy sampler by default.", magic_enum::enum_name(sampler_type));
            }

            return std::make_unique<GreedySampler>();
        }

    };
    
} // namespace spy