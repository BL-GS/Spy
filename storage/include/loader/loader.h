#pragma once

#include <cstdint>
#include <string_view>
#include <span>

#include "adapter/type.h"

namespace spy {

    class ModelLoader {
    public:
        ModelMetaContext context;

    public:
        ModelLoader() = default;

        virtual ~ModelLoader() noexcept = default;

    public:
        /*!
         * @brief [optional] Preload the file into the memory to accelerate consequent data loading
         */
        virtual void preload() = 0;

        /*!
         * @brief Load a specific data into the memory from file
         * @param[in] name The name of data
         * @return The memory range of the data
         */
        virtual std::span<uint8_t> load(std::string_view name) = 0;
    };

    struct ModelLoaderFactory {
        static std::unique_ptr<ModelLoader> build_model_loader(std::string_view policy, std::string_view filename);
    };

} // namespace spy