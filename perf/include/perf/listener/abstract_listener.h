#pragma once

#include <vector>
#include <string>
#include <string_view>

namespace spy::perf {

    struct ProfileRecord {
    public:
        struct KVPair {
            std::string_view key;
            std::string      value;
        };

    public:
        std::vector<KVPair> kv_pairs;

    public:
        void operator +=(const ProfileRecord &other) {
            for (const KVPair &kv_pair: other.kv_pairs) {
                kv_pairs.emplace_back(kv_pair);
            }
        }

        ProfileRecord operator +(const ProfileRecord &other) const {
            ProfileRecord new_record = *this;
            new_record += other;
            return new_record;
        }
    };

    class AbstractProfilerListener {
    public:
        AbstractProfilerListener() = default;

        virtual ~AbstractProfilerListener() noexcept = default;

    public:
        virtual ProfileRecord get_hardware_info() const { return {}; }

    public:
        virtual void start() = 0;

        virtual ProfileRecord profile() = 0;
    };

} // namespace spy::perf