#include <string>
#include <magic_enum.hpp>

#include "graph/op_node.h"

namespace spy {

    std::map<std::string_view, std::string> OperatorNode::property() const {
        std::map<std::string_view, std::string> basic = BasicNode::property();

        basic["operator"] = magic_enum::enum_name(op_type);
        return basic;
    }

} // namespace spy