/*
 * @author: BL-GS
 * @date:   24-4-13
 */

#include <magic_enum_switch.hpp>
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <unistd.h>
#endif // _WIN32

#include "operator_impl.h"
#include "abstract_backend.h"
#include "task.h"

namespace spy::cpu {

#ifdef _WIN32
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
#else 

    size_t CPUBackend::get_max_memory_capacity() const {
		const size_t num_page  = sysconf(_SC_PHYS_PAGES);
		const size_t page_size = sysconf(_SC_PAGE_SIZE);
		return page_size * num_page;
    }

    size_t CPUBackend::get_avail_memory_capacity() const {
	    const size_t num_page  = sysconf(_SC_AVPHYS_PAGES);
	    const size_t page_size = sysconf(_SC_PAGE_SIZE);
	    return page_size * num_page;
    }
    
    size_t CPUBackend::get_max_concurrency() const {
        return sysconf( _SC_NPROCESSORS_CONF);
    }

    size_t CPUBackend::get_avail_concurrency() const {
        return sysconf(_SC_NPROCESSORS_ONLN);
    }

#endif // _WIN32

    OperatorResult CPUBackend::execute(const OperatorEnvParam &param, OperatorNode *node_ptr) {
        const OperatorType op_type = node_ptr->op_type;

        const auto func = get_execute_func(op_type);
        return func(this, param, node_ptr);
    }

    CPUBackend::TaskFunc CPUBackend::get_execute_func(OperatorType op_type) const {
        return magic_enum::enum_switch([](auto type){
            return cpu::OperatorImpl<type>::execute;
        }, op_type);
    }

    std::shared_ptr<ControlHeader> CPUBackend::get_control_header(OperatorType op_type, OperatorNode *node_ptr) {
        return magic_enum::enum_switch([this, node_ptr](auto type){
            return cpu::OperatorImpl<type>::get_control_header(this, node_ptr);
        }, op_type);
    }

} // namespace spy

