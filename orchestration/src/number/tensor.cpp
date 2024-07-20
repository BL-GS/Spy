#include <map>
#include <string_view>
#include <string>
#include <magic_enum.hpp>

#include "number/tensor.h"

namespace spy {

    template<class T, size_t N>
    inline static std::string array_to_string(const std::array<T, N> &arr, size_t num) {
        std::string res;
        for (size_t i = 0; i < num; ++i) {
            if (i != 0) { res += ", "; }
            res += std::to_string(arr[i]);
        }
        return res;
    }

    std::string Shape::to_string() const {
        return fmt::format("({}) ({})", 
            array_to_string(elements, dim), array_to_string(bytes, dim));
    }

    std::map<std::string_view, std::string> Shape::property() const {
        std::map<std::string_view, std::string> property = {
            { "number type", std::string(magic_enum::enum_name(number_type)) },
            { "dim"        , std::to_string(dim)                   },
            { "#element"   , array_to_string(elements, dim)   },
            { "#byte"      , array_to_string(bytes, dim)      }
        };
        return property;
    }

    std::map<std::string_view, std::string> Tensor::property() const {
        return shape.property();
    }

} // namespace spy