#pragma once

#include <cstdio>
#include <cstddef>
#include <string_view>

#include "util/unit.h"
#include "util/file.h"
#include "util/logger.h"
#include "model/file/config.h"

namespace spy {
    
    class ModelMapper {
    public:
        SpyFile                 file;
        FileMappingViewFactory  file_view_factory;

    public:
        ModelMapper(const std::string_view filename): file(filename, "rb") {
            file_view_factory.set_fd(file.get_fp(), file.get_fd());
            SPY_INFO_FMT("Open file: {} (size: {} MB)", filename, file.size() / 1_MB);
        }

        ~ModelMapper() noexcept = default;

        ModelMapper(ModelMapper &&other) noexcept = default;

    public:
        void mapping(GGUFContext &context) {
            const size_t data_offset = context.offset;
            const size_t data_size   = context.size;
            if (data_offset + data_size > file.size()) { 
                SPY_WARN_FMT("Trying to map file with excessive size: (map: {}, file: {})", data_offset + data_size, file.size()); 
            }
            SPY_INFO_FMT("Mapping data section (offset: 0x{:x}, size: {}MB)", data_offset, data_size / 1_MB);
            file_view_factory.create_view(data_offset, data_size);

            for (auto &info_pair: context.infos) {
                auto &info = info_pair.second;
                const size_t cur_offset = info.offset + data_offset;

                SPY_ASSERT_FMT(cur_offset >= data_offset && cur_offset < data_offset + data_size, 
                    "The offset of view(0x{:x}) should be within the data section(0x{:x} - 0x{:x})",
                    cur_offset, context.offset, context.offset + context.size);
                // Assign data point 
                info.data_ptr = file_view_factory.get_view(context.offset)->deref(cur_offset);
            }
        }
    };

} // namespace spy