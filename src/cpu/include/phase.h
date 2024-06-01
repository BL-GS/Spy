#pragma once

#include <cstdint>

namespace spy::cpu {

        /// 
    /// @brief Phases of operator
    /// There should be thread barrier between different phases except for 1 threads -> all threads
    /// The phase property should only be written by the main thread (tid == 0)
    /// 
    enum class OperatorPhaseType: uint32_t {
        /// Initialization
        /// Accessible to the main thread
        Init       = 0x0000'0000,
        /// Preparation
        /// Accessible to all threads
        Prepare    = 0x0001'0000,
        /// Execution
        /// Accessible to all threads
        Execute    = 0x0002'0000,
        /// Release all holded resources
        /// Accessible to all threads
        Release    = 0x0003'0000,
        /// Finsh
        /// Accessible to the main thread
        Finish     = 0x0004'0000,
        /// End of operator
        End        = 0x0005'0000
    };

    inline static constexpr OperatorPhaseType get_base_operator_phase(OperatorPhaseType phase) {
        const     uint32_t phase_num = static_cast<uint32_t>(phase);
        constexpr uint32_t base_mask = 0xFFFF'0000;
        const     uint32_t phase_id  = phase_num & base_mask;
        return static_cast<OperatorPhaseType>(phase_id);
    }

    inline static constexpr OperatorPhaseType operator_phase(OperatorPhaseType base, uint32_t id) {
        return static_cast<OperatorPhaseType>(static_cast<uint32_t>(base) + id);
    }

    inline static constexpr OperatorPhaseType operator""_op_init(unsigned long long id)     { return operator_phase(OperatorPhaseType::Init, id);       }
    inline static constexpr OperatorPhaseType operator""_op_prepare(unsigned long long id)  { return operator_phase(OperatorPhaseType::Prepare, id);    }
    inline static constexpr OperatorPhaseType operator""_op_execute(unsigned long long id)  { return operator_phase(OperatorPhaseType::Execute, id);    }
    inline static constexpr OperatorPhaseType operator""_op_release(unsigned long long id)  { return operator_phase(OperatorPhaseType::Release, id);    }
    inline static constexpr OperatorPhaseType operator""_op_finish(unsigned long long id)   { return operator_phase(OperatorPhaseType::Finish, id);     }
    inline static constexpr OperatorPhaseType operator""_op_end(unsigned long long id)      { return operator_phase(OperatorPhaseType::End, id);        }

} // namepsace spy::cpu