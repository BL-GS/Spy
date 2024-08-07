#include <memory>

#define BACKEND_DISTRIBUTOR_HEADER_MACRO

#include "util/log/logger.h"
#include "distributor/distributor.h"
#include "distributor/simple_distributor.h"

namespace spy {

    std::unique_ptr<GraphDistributor> GraphDistributorFactory::build_graph_distributor(std::string_view policy, ModelLoader *loader_ptr) {

        if (policy == "simple") {
            return std::make_unique<SimpleGraphDistributor>(loader_ptr);
        }

        spy_error("unknown kind of graph distributor: {}", policy);
        return nullptr;
    }

} // namespace spy