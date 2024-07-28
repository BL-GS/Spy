#pragma once

#include <cstdint>
#include <string_view>
#include <span>

#include "loader/file/file_view.h"
#include "loader/file/mapper.h"
#include "loader/loader.h"

namespace spy {

    class SimpleModeLoader final: public ModelLoader {
    public:
        FileViewBuilder  builder;

	    FileMappingView  view_buffer;

    public:
        SimpleModeLoader(std::string_view filename);

        ~SimpleModeLoader() noexcept override = default;

    public:
        void preload() override;

        std::span<uint8_t> load(std::string_view name) override;
    };

} // namespace spy