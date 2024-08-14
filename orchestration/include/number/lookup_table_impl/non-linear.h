#pragma once

#include <cstdint>
#include <cmath>
#include <array>
#include <simde/simde-f16.h>

#include "number/lookup_table_impl/type.h"

namespace spy {

    enum class NonLinearLookupTableType {
        Silu, Gelu, Exp
    };

    template<NonLinearLookupTableType T_type, NumberType T_from_type, NumberType T_to_type>
    struct NonLinearLookupTable { };

    template<>
    struct NonLinearLookupTable<NonLinearLookupTableType::Silu, NumberType::FP16, NumberType::FP32> { 
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
        std::array<Value, NUM_ENTRY> table;

    public:
        NonLinearLookupTable() {
            for (Key k = 0; k < MAX_KEY; ++k) { 
                const float float_key = simde_float16_to_float32(simde_uint16_as_float16(k));
                table[k] = float_key / (1.0F + std::exp(-float_key)); 
            }
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


    template<NumberType T_from_type, NumberType T_to_type>
    using SiluFP16LookupTable = NonLinearLookupTable<NonLinearLookupTableType::Silu, T_from_type, T_to_type>;
    

    template<>
    struct NonLinearLookupTable<NonLinearLookupTableType::Gelu, NumberType::FP16, NumberType::FP32> { 
    public:
        static constexpr NumberType FromType  = NumberType::FP16;
        static constexpr NumberType ToType    = NumberType::FP32;

        using Key       = BlockType<FromType>;
        using Value     = BlockType<ToType>;

        static constexpr Key    MIN_KEY = std::numeric_limits<Key>::min();
        static constexpr Key    MAX_KEY = std::numeric_limits<Key>::max();
        static constexpr Value  MIN_VAL = std::numeric_limits<Value>::min();
        static constexpr Value  MAX_VAL = std::numeric_limits<Value>::max();

        static constexpr size_t NUM_ENTRY = MAX_KEY - MIN_KEY + 1;

    public:
        std::array<Value, NUM_ENTRY> table;

    public:
        NonLinearLookupTable() {
            constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876F;
			constexpr float GELU_COEF_A    = 0.044715F;
            for (Key k = 0; k < MAX_KEY; ++k) { 
                const float float_key = simde_float16_to_float32(simde_uint16_as_float16(k));
                table[k] = 0.5F * float_key * (1.0F + tanhf(SQRT_2_OVER_PI * float_key * (1.0F + GELU_COEF_A * float_key * float_key))); 
            }
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


    template<NumberType T_from_type, NumberType T_to_type>
    using GeluLookupTable = NonLinearLookupTable<NonLinearLookupTableType::Gelu, T_from_type, T_to_type>;

    template<>
    struct NonLinearLookupTable<NonLinearLookupTableType::Exp, NumberType::FP16, NumberType::FP32> { 
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
        std::array<Value, NUM_ENTRY> table;

    public:
        NonLinearLookupTable() {
            for (Key k = 0; k < MAX_KEY; ++k) { 
                const float float_key = simde_float16_to_float32(simde_uint16_as_float16(k));
                table[k] = std::exp(float_key); 
            }
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


    template<NumberType T_from_type, NumberType T_to_type>
    using ExpLookupTable = NonLinearLookupTable<NonLinearLookupTableType::Exp, T_from_type, T_to_type>;

} // namespace spy