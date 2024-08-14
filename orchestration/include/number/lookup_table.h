#pragma once

#include <cstdint>

#include "number/number.h"
#include "number/lookup_table_impl/type.h"
#include "number/lookup_table_impl/conversion.h"
#include "number/lookup_table_impl/non-linear.h"

namespace spy {    

    struct FP16LookupTable {
	public:
        using FP32Table = ConversationLookupTable<NumberType::FP16, NumberType::FP32>;
        using SiluTable = SiluFP16LookupTable<NumberType::FP16, NumberType::FP32>;
        using GeluTable = GeluLookupTable<NumberType::FP16, NumberType::FP32>;
        using ExpTable  = ExpLookupTable<NumberType::FP16, NumberType::FP32>;

	public:
        FP32Table fp32_table;
        SiluTable silu_table;
        GeluTable gelu_table;
        ExpTable  exp_table;

	public:
        FP16LookupTable() = default;

	public:
		/*!
		 * @brief Convert fp16 value into fp32 value
		 */
	    float fp32(uint16_t val) const { return fp32_table(val);    }

		/*!
		 * @brief Lookup the gelu result of the fp16 argument
		 */
		float gelu(uint16_t val) const { return gelu_table(val);    }
		
		/*!
		 * @brief Lookup the silu result of the fp16 argument
		 */
		float silu(uint16_t val) const { return silu_table(val);    }

		/*!
		 * @brief Lookup the exp result of the fp16 argument
		 */
		float exp(uint16_t val)  const { return exp_table(val);     }
	};
	
	extern const FP16LookupTable LOOK_UP_TABLE;

	inline float spy_fp16_to_fp32(uint16_t x) { return LOOK_UP_TABLE.fp32(x); }

	inline uint16_t spy_fp32_to_fp16(float x) { return simde_float16_as_uint16(simde_float16_from_float32(x)); }
    
} // namespace spy