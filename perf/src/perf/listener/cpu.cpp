#include <cstdint>
#include <cstdio>
#include <thread>

#include "perf/listener/exception.h"
#include "perf/listener/cpu.h"

namespace spy::perf {

#ifdef _WIN32

    ProfileRecord ProcessorProfiler::get_hardware_info() const { return {}; }

    void ProcessorProfiler::start() { }

    ProfileRecord ProcessorProfiler::profile() { return {}; }

#elif __linux__

    static constexpr std::string_view STAT_FILENAME = "/proc/stat";

    static ProcessorProfilerInfo get_processor_counter() {
        struct LinuxCPUCount {
            char name[20]; 
            unsigned int user;
            unsigned int nice;
            unsigned int system;
            unsigned int idle;
            unsigned int lowait;  
            unsigned int irq;  
            unsigned int softirq;  
        } cpu_info;

        char buffer[256];
        FILE *fd = fopen(STAT_FILENAME.data(), "r");  
        if (fd == nullptr) { throw SpyPerfException("failed reading file: {}", STAT_FILENAME); }

        fgets(buffer, sizeof(buffer), fd);  
        sscanf(buffer, "%s %u %u %u %u %u %u %u", cpu_info.name, &cpu_info.user, &cpu_info.nice, &cpu_info.system, &cpu_info.idle, &cpu_info.lowait, &cpu_info.irq, &cpu_info.softirq);  
        fclose(fd);  

        return {
            .total  = cpu_info.user + cpu_info.system + cpu_info.nice + cpu_info.lowait + cpu_info.irq + cpu_info.softirq,
            .user   = cpu_info.user,
            .system = cpu_info.system,
            .io     = cpu_info.lowait,
            .idle   = cpu_info.idle
        };
    }

    static ProfileRecord processor_info_to_record(const ProcessorProfilerInfo &info) {
        return {{
            { "user", std::to_string(info.user * 100 / info.total) },
            { "system", std::to_string(info.system * 100 / info.total) },
            { "io", std::to_string(info.io * 100 / info.total) },
            { "idle", std::to_string(info.idle * 100 / info.total) },
        }};
    }

    ProfileRecord ProcessorProfiler::get_hardware_info() const {
        return {{
            { "num_processor", std::to_string(std::thread::hardware_concurrency()) }
        }};
    }

    void ProcessorProfiler::start() { 
        prev_info_ = std::make_unique<ProcessorProfilerInfo>(get_processor_counter());
    }

    ProfileRecord ProcessorProfiler::profile() {
        const ProcessorProfilerInfo new_info = get_processor_counter();
        const ProcessorProfilerInfo res_info = new_info - *prev_info_;
        *prev_info_ = new_info;
        return processor_info_to_record(res_info);
    }

#endif // _WIN32 or __linux__

} // namespace spy