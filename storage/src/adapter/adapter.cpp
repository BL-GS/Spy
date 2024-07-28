#include <memory>
#include <string_view>
#include <filesystem>

#include "loader/file/exception.h"
#include "adapter/abstract_adapter.h"
#include "adapter/gguf_adaper.h"
#include "adapter/adapter.h"

namespace spy {

    using FileAdapterPointer = std::unique_ptr<AbstractFileAdapter>;

    FileAdapterPointer FileAdapterFactory::build_file_adapter(std::string_view type) {
        if (type == "gguf") {
            return std::make_unique<GGUFAdapter>();
        }
        return nullptr;
    }

    FileAdapterPointer FileAdapterFactory::auto_build_file_adapter(std::filesystem::path path) {
        const std::string path_str = path.string();

        if (!std::filesystem::exists(path)) {
            throw SpyOSFileException("the path to the model does not exist: {}", path_str);
        }

        if (std::filesystem::is_directory(path)) {

        } else if (std::filesystem::is_regular_file(path)) {
            if (path.has_extension()) { 
                std::string ext = path.extension().string();

                if (ext == ".gguf") {
                    // TODO: test adapter
                    return std::make_unique<GGUFAdapter>();
                }
            }

            spy_warn("the filename({}) does not has extension. It may spend a lot of time to get a proper adapter", 
                path.filename().c_str()); 
        }

        throw SpyOSFileException("unsupported file type, expect as directory or regular file: {}", path_str);
    }

} // namespace

