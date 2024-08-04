#include <cstring>

#ifdef WIN32
    #include <pdh.h>
    #include <psapi.h>
    #include <tchar.h>
    #pragma comment(lib, "pdh")
#elif __linux__
    #include <sys/sysinfo.h>
    #include <sys/times.h>
    #include <sys/types.h>
#endif

#include "perf/listener/exception.h"
#include "perf/listener/mem.h"

#ifdef SPY_PERFETTO_TRACING
    #include "perfetto.h"
    #include "perf/event.h"
#endif // SPY_PERFETTO_TRACING

namespace spy::perf {

#ifdef _WIN32

    ProfileRecord MemoryProfiler::get_hardware_info() const { return {}; }

    void MemoryProfiler::start() { /* nothing to do */ }

    ProfileRecord MemoryProfiler::profile() { return {}; }

#elif __linux__

    static constexpr std::string_view STATUS_FILENAME = "/proc/self/status";
    static constexpr std::string_view STAT_FILENAME   = "/proc/stat";

    struct MemoryProfilerInfo {
        int64_t total_virtual = 0;
        int64_t avail_virtual = 0;
        int64_t process_used_virtual  = 0;
        int64_t total_physical = 0;
        int64_t avail_physical = 0;
        int64_t process_used_physical = 0;
    };

    inline static int64_t parse_mem_status_line(char *line) {
        // This assumes that a digit will be found and the line ends in " Kb".
        int i = strlen(line);
        const char *p = line;
        while (*p < '0' || *p > '9') { p++; }
        line[i - 3] = '\0';
        return atoll(p) * 1000; // translate to bytes
    }

    static MemoryProfilerInfo get_memory_info() {

        struct sysinfo mem_info;
        sysinfo(&mem_info);

        int64_t total_physical = mem_info.totalram;
        // Multiply in next statement to avoid int overflow on right hand side...
        total_physical *= mem_info.mem_unit;

        int64_t used_physical = mem_info.totalram - mem_info.freeram;
        // Multiply in next statement to avoid int overflow on right hand side...
        used_physical *= mem_info.mem_unit;

        MemoryProfilerInfo info;
        info.total_physical = used_physical;
        info.avail_physical = total_physical;
        info.process_used_physical = 0;

        int64_t total_virtual = mem_info.totalram;
        // Add other values in next statement to avoid int overflow on right hand side...
        total_virtual += mem_info.totalswap;
        total_virtual *= mem_info.mem_unit;

        int64_t used_virtual = mem_info.totalram - mem_info.freeram;
        // Add other values in next statement to avoid int overflow on right hand side...
        used_virtual += mem_info.totalswap - mem_info.freeswap;
        used_virtual *= mem_info.mem_unit;

        info.total_virtual = total_virtual;
        info.avail_virtual = total_virtual - used_virtual;
        info.process_used_virtual = 0;

        FILE *file = fopen(STATUS_FILENAME.data(), "r");
        if (file == NULL) {
            throw SpyPerfException("failed reading file: {}", STATUS_FILENAME);
        }

        char line[128];
        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmSize:", 7) == 0) {
                info.process_used_virtual = parse_mem_status_line(line);
            } else if (strncmp(line, "VmRSS:", 6) == 0) {
                info.process_used_physical = parse_mem_status_line(line);
            }
        }
        fclose(file);

        return info;
    }

    ProfileRecord MemoryProfiler::get_hardware_info() const {
        const MemoryProfilerInfo info = get_memory_info();
        return {{
            { "total_virtual_memory", std::to_string(info.total_virtual) },
            { "total_physical_memory", std::to_string(info.total_physical) },
        }};
    }

    void MemoryProfiler::start() { /* nothing to do */ }

    ProfileRecord MemoryProfiler::profile() {
        const MemoryProfilerInfo info = get_memory_info();

#ifdef SPY_PERFETTO_TRACING
        using Unit = perfetto::CounterTrack::Unit;
        {
            perfetto::CounterTrack physical_memory_track = perfetto::CounterTrack("PhysicalMemory").set_unit(Unit::UNIT_SIZE_BYTES);
            TRACE_COUNTER(SYSTEM_CATEGORY.data(), physical_memory_track, info.process_used_physical);
            perfetto::CounterTrack virtual_memory_track = perfetto::CounterTrack("VirtualMemory").set_unit(Unit::UNIT_SIZE_BYTES);
            TRACE_COUNTER(SYSTEM_CATEGORY.data(), virtual_memory_track, info.process_used_physical);            
        }
#endif

        return {{
            { "avail_virtual_memory", std::to_string(info.avail_virtual) },
            { "process_virtual_memory", std::to_string(info.process_used_virtual) },
            { "avail_physical_memory", std::to_string(info.avail_physical) },
            { "process_physical_memory", std::to_string(info.process_used_physical) },
        }};
    }

#endif // _WIN32 or __linux__

} // spy::perf