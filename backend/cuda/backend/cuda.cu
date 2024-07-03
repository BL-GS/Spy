
#include <memory>

#include "util/shell/logger.h"
#include "backend/config.h"
#include "default_backend.h"

namespace spy::gpu {

    void init_backend(BackendFactory &factory) {
        using BackendConfiguration = BackendFactory::BackendConfiguration;

        spy_info("Register CPU backend");

        const std::string default_backend_name = BackendFactory::make_backend_name("gpu", "default");
        factory.add_backend_map(default_backend_name, 
            +[](const BackendConfiguration &config) -> std::unique_ptr<AbstractBackend> {
                return std::make_unique<DefaultGPUBackend>(config);
            }
        );        
    }

} // namespace spy::gpu