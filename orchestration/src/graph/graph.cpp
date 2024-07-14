#include <string>
#include <string_view>
#include <map>

#include "graph/graph.h"

namespace spy {

    std::map<std::string_view, std::string> Graph::property() const {
        return {
            { "id",     std::to_string(id) },
            { "entry",  entry_point->name       }
        };
    }

} // namespace spy