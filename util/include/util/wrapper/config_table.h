#pragma once

#include <map>
#include <optional>
#include <string>
#include <charconv>

namespace spy {

    class ConfigTable {
    public:
        using MapType = std::map<std::string, std::string>;

    private:
        MapType table_;

    public:
        ConfigTable() = default;

        template<class ...Args>
        ConfigTable(Args &&...args): table_(std::forward<Args>(args)...) {}

        ConfigTable(const ConfigTable &other) = default;

        ConfigTable(ConfigTable &&other) = default;

    public:
        void add(const std::string_view key, const std::string_view value) {
            table_[std::string(key)] = value;
        }

        void remove(const std::string_view key) {
            table_.erase(std::string(key));
        }

    public:
        std::optional<std::string> get(const std::string_view key) const {
            const auto iter = table_.find(std::string(key));
            if (iter == table_.cend()) { return std::nullopt; }
            return iter->second;
        }

        std::string get_or(const std::string_view key, const std::string &default_value) const {
            const auto iter = table_.find(std::string(key));
            if (iter == table_.cend()) { return default_value; }
            return iter->second;
        }

        template<class T>
            requires std::is_integral_v<T> || std::is_floating_point_v<T>
        std::optional<T> parse(const std::string_view key) const {
            const auto iter = table_.find(std::string(key));
            if (iter == table_.cend()) { return std::nullopt; }
            const std::string &value = iter->second;

            T res;
            std::from_chars(value.begin(), value.end(), res);
            return res;
        }

        template<class T>
            requires std::is_integral_v<T> || std::is_floating_point_v<T>
        T parse_or(const std::string_view key, const T default_value) const {
            const auto iter = table_.find(std::string(key));
            if (iter == table_.cend()) { return default_value; }
            const std::string &value = iter->second;

            T res = 0;
            std::from_chars(value.data(), value.data() + value.size(), res);
            return res;
        }
    };

} // namespace spy