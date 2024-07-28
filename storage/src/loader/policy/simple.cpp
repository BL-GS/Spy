#include <cstdint>
#include <string_view>
#include <span>

#include "util/unit.h"
#include "adapter/adapter.h"
#include "loader/policy/simple.h"

namespace spy {
    
    SimpleModeLoader::SimpleModeLoader(std::string_view filename) {
        {
            {
                auto file_adapter_ptr = FileAdapterFactory::auto_build_file_adapter(filename);
                context = file_adapter_ptr->init_from_file(filename);                
            }

			builder.init_sync_handle(filename);
			builder.init_mapping();
		}
    }

    void SimpleModeLoader::preload() {
        const size_t data_offset = context.offset;
        const size_t data_size   = context.size;
        spy_info("Loading data section (offset: 0x{:x}, size: {}MB)", data_offset, data_size / 1_MB);

        try {
            FileMappingView view = builder.create_mapping_view(data_size, data_offset);

            for (auto &info_pair: context.infos) {
                auto &info = info_pair.second;
                // Assign data point
                info.data_ptr = view.deref(info.offset);
            }

            view_buffer = std::move(view);
        } catch (...) {
            spy_error("Failed loading model");
            throw;
        }
    }

    std::span<uint8_t> SimpleModeLoader::load(std::string_view name) try {
        const auto data_info = context.infos.at(std::string(name));

        uint8_t *data_ptr  = static_cast<uint8_t *>(data_info.data_ptr);
        size_t   num_bytes = get_type_size(data_info.type);
        for (int i = 0; i < data_info.num_dim; ++i) { num_bytes *= data_info.num_element[i]; }

        return { data_ptr, num_bytes };
    } catch (std::out_of_range &err) {
        spy_fatal("data unexists: {}", name);
        throw err;
    }

} // namespace spy