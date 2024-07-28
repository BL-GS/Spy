#pragma once

#include <cstdio>
#include <cstddef>
#include <string_view>
#include <memory>

#include "util/unit.h"
#include "util/shell/logger.h"
#include "file/file_view.h"
#include "file/mapper.h"
#include "adapter/adapter.h"

namespace spy {
    
    class ModelLoader {
    public:
        ModelMetaContext                     context;

        FileViewBuilder                      builder;

		std::vector<FileMappingView>         view_buffer;

    public:
        ModelLoader(const std::string_view filename) {
            {
                auto file_adapter_ptr = FileAdapterFactory::auto_build_file_adapter(filename);
                context = file_adapter_ptr->init_from_file(filename);                
            }

			builder.init_sync_handle(filename);
			builder.init_mapping();
		}

        ~ModelLoader() noexcept = default;

    public:
        size_t load() {
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

				view_buffer.emplace_back(std::move(view));
            } catch (...) {
                spy_error("Failed loading model");
                throw;
            }

            return data_size;
        }
        
    };

} // namespace spy