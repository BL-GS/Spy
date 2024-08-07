/*
 * @author: BL-GS 
 * @date:   24-4-13
 */

#include <memory>

#include "util/log/logger.h"
#include "backend/config.h"
#include "default_backend.h"

namespace spy::cpu {

    void init_backend(BackendFactory &factory) {
        using BackendConfiguration = BackendFactory::BackendConfiguration;

        spy_info("Register CPU backend");

        const std::string default_backend_name = BackendFactory::make_backend_name("cpu", "default");
        factory.add_backend_map(default_backend_name, 
            +[](const BackendConfiguration &config) -> std::unique_ptr<Backend> {
                return std::make_unique<DefaultCPUBackend>(config);
            }
        );        
    }

} // namespace spy::cpu