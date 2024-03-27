/*
 * @author: BL-GS
 * @date:   24-4-13
 */

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#include "backend/cpu/type.h"

namespace spy {

    size_t CPUBackend::get_max_memory_capacity() const {
        MEMORYSTATUS mem_status;
        GlobalMemoryStatus(&mem_status);
        return mem_status.dwTotalPhys;
    }

    size_t CPUBackend::get_avail_memory_capacity() const {
        MEMORYSTATUS mem_status;
        GlobalMemoryStatus(&mem_status);
        return mem_status.dwAvailPhys;
    }
    
    size_t CPUBackend::get_max_concurrency() const {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        return sys_info.dwNumberOfProcessors;
    }

    size_t CPUBackend::get_avail_concurrency() const {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        return sys_info.dwNumberOfProcessors;
    }


} // namespace spy

#endif // _WIN32