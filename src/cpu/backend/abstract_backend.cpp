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

    size_t CPUBackend::get_task_num(const OperatorNode *node_ptr) const {
        const OperatorType op_type = node_ptr->op_type;

        auto func = magic_enum::enum_switch([](auto type){
            return cpu::OperatorImpl<type>::get_task_num;
        }, op_type);
        return func(this, node_ptr);
    }

    size_t CPUBackend::get_buffer_size(const OperatorNode *node_ptr) const {
        const OperatorType op_type = node_ptr->op_type;

        auto func = magic_enum::enum_switch([](auto type){
            return cpu::OperatorImpl<type>::get_buffer_size;
        }, op_type);
        return func(this, node_ptr);
    }

    OperatorStatus CPUBackend::execute(const OperatorEnvParam &param, OperatorNode *node_ptr) {
        const OperatorType op_type = node_ptr->op_type;

        auto func = magic_enum::enum_switch([](auto type){
            return cpu::OperatorImpl<type>::execute;
        }, op_type);
        return func(this, param, node_ptr);
    }

} // namespace spy

