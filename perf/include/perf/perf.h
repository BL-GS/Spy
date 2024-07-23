#pragma once

#include <vector>

#include "util/shell/logger.h"
#include "perf/listener/listener.h"

namespace spy::perf {

    class PerformanceProfiler {
    private:
        std::vector<std::unique_ptr<AbstractProfilerListener>> listener_list_;

    public:
        PerformanceProfiler(const std::string &name);
        
    public:
        void add_listener(ListenerType type) {
            try {
                listener_list_.emplace_back(ListenerFactory::build_listener(type));
            } catch (SpyPerfException &err) {
                spy_error("failed adding listener: {}", err.what());
            }
        }

        void start() {
            for (auto &listener_ptr: listener_list_) {
                listener_ptr->start();
            }
        }

        ProfileRecord trigger() {
            ProfileRecord record;
            for (auto &listener_ptr: listener_list_) {
                record += listener_ptr->profile();
            }
            return record;
        }
    };

} // namespace spy