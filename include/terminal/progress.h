#pragma once

#include <string>
#include <string_view>

#include "util/shell/logger.h"

namespace spy {

    struct ProgressBar {
    public:
        static constexpr std::string_view EMPTY_PROGRESS_BAR = "[--------------------]";
        static constexpr std::string_view PROGRESS_BAR_FMT   = "{} {:4}%";
        static constexpr size_t           MAX_BAR_LEN        = EMPTY_PROGRESS_BAR.size() - 2;

    public:
        size_t      max_progress;
        size_t      progress;
        size_t      percentage;
        std::string bar;

    public:
        ProgressBar(size_t max_progress): max_progress(max_progress), progress(0), percentage(0), bar(EMPTY_PROGRESS_BAR) {}

    public:
        bool step() {
            ++progress;
            size_t new_percentage = get_percentage();
            if (new_percentage != percentage) {
                percentage = new_percentage;
                return true;
            }
            return false;
        }

        void reset() {
            progress = 0;
        }

        std::string to_string() {
            update_bar();
            return fmt::format(PROGRESS_BAR_FMT, bar, percentage);
        }

        void output() {
            std::cout << to_string() << '\r';
        }

    private:
        void update_bar() {
            const size_t bar_len = get_bar_len();
            for (size_t i = 1; i <= bar_len; ++i) { bar[i] = '='; }
            for (size_t i = progress + 1; i <= MAX_BAR_LEN; ++i) { bar[i] = '-'; }
        }

        size_t get_bar_len() const {
            return std::min(MAX_BAR_LEN * progress / max_progress, MAX_BAR_LEN);
        }

        size_t get_percentage() const {
            return progress * 100 / max_progress;
        }

    };

    template<class Iter, class Func>
    inline void progress(Iter start, Iter end, Func &&func) {
        size_t total_progress = end - start;
        ProgressBar progress_bar(total_progress);

        for (auto iter = start; iter < end; ++iter) { 
            if (progress_bar.step()) { progress_bar.output(); }
            func(iter); 
        }
    }

} // namespace spy