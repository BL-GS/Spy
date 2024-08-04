#include <memory>

#include "util/log/logger.h"
#include "distributor/distributor.h"
#include "distributor/policy/simple.h"

namespace spy {

    std::unique_ptr<AbstractGraphDistributor> GraphDistributorFactory::build_graph_distributor(std::string_view policy, ModelLoader *loader_ptr) {

        if (policy == "simple") {
            return std::make_unique<SimpleGraphDistributor>(loader_ptr);
        }

        spy_error("unknown kind of graph distributor: {}", policy);
        return nullptr;
    }

} // namespace spy