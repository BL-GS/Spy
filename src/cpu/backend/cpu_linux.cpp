/*
 * @author: BL-GS
 * @date:   24-4-13
 */

#ifdef __linux__

#include <unistd.h>

#include "backend/cpu/type.h"

namespace spy {

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

} // namespace spy

#endif // __linux__