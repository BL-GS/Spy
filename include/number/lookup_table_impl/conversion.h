#pragma once

#include <cstdint>
#include <array>

#include "number/number.h"

/* 
 * Lookup table for number conversation
 */

namespace spy {

    template<NumberType T_from_type, NumberType T_to_type>
    struct ConversationLookupTable { };

    template<>
    struct ConversationLookupTable<NumberType::FP16, NumberType::FP32> { 
    public:
        static constexpr NumberType FromType  = NumberType::FP16;
        static constexpr NumberType ToType    = NumberType::FP32;

        using Key       = BlockType<FromType>;
        using Value     = BlockType<ToType>;

        static_assert(std::is_same_v<Key,   uint16_t>);
        static_assert(std::is_same_v<Value, float>);

        static constexpr Key    MIN_KEY = std::numeric_limits<Key>::min();
        static constexpr Key    MAX_KEY = std::numeric_limits<Key>::max();
        static constexpr Value  MIN_VAL = std::numeric_limits<Value>::min();
        static constexpr Value  MAX_VAL = std::numeric_limits<Value>::max();

        static constexpr size_t NUM_ENTRY = MAX_KEY - MIN_KEY + 1;

    public:
        std::array<float, NUM_ENTRY> table;

    public:
        ConversationLookupTable() {
            for (Key k = 0; k < MAX_KEY; ++k) { table[k] = spy_fp16_to_fp32(k); }
        }

    public:
        Value lookup(const Key key)         const { return table[key]; }

        Value operator() (const Key key)    const { return lookup(key); }

    public:
        static constexpr Key   min_key()  { return MIN_KEY; }
        static constexpr Key   max_key()  { return MAX_KEY; }
        static constexpr Value min_val()  { return MIN_VAL; }
        static constexpr Value max_val()  { return MAX_VAL; }
    };
    
} // namespace spy