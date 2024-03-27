#pragma once

#include "number/number.h"

namespace spy {

    template<class T_LookupTable, NumberType T_from_type = T_LookupTable::FromType, NumberType T_to_type = T_LookupTable::ToType>
    concept LookupTableConcept = requires(T_LookupTable table, BlockType<T_from_type> val) {
        /// Lookup functions
        { table.lookup(val)     }   -> std::same_as<BlockType<T_to_type>>;
        { table.operator()(val) }   -> std::same_as<BlockType<T_to_type>>;
        /// Constraint of key
        { table.min_key()       }   -> std::same_as<BlockType<T_from_type>>;
        { table.max_key()       }   -> std::same_as<BlockType<T_from_type>>;
        /// Constraint of value
        { table.min_val()       }   -> std::same_as<BlockType<T_to_type>>;
        { table.max_val()       }   -> std::same_as<BlockType<T_to_type>>;
    };

} // namespace spy