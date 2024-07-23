#include <fstream>
#include <numeric>

#include "perf/perf.h"

namespace spy {

    inline constexpr std::string_view CSV_SUFFIX = ".csv";

    class AppendOnlyCsv {
    private:
        size_t                                num_column_;
        std::ofstream                         output_file_;

    public:
        AppendOnlyCsv(std::string filename, std::vector<std::string> header): 
            num_column_(header.size()) {

            spy_assert(num_column_ > 0, 
                "Expect the number of column to be larger than 1 (get: {})", num_column_);

            filename += CSV_SUFFIX;
            spy_info("Create append-only csv file: {}", filename);
            output_file_.open(filename, std::ios_base::out | std::ios_base::ate | std::ios_base::trunc);
            if (!output_file_.good()) {
                throw SpyOSException("Failed opening file: " + filename);
            }

            add_row(header);
        }

        ~AppendOnlyCsv() noexcept = default;

    public:
        void add_row(const std::vector<std::string> &row) {
            spy_assert(row.size() == num_column_, 
                "Expect the number of columns in a row equals to that of header: {} (get: {})", 
                num_column_, row.size()
            );

            std::string row_str = std::reduce(row.begin(), row.end(), std::string(), 
                [](auto &a, auto &b){ return a + ',' + b; });
            // Replace the last ',' with '\n'
            *row_str.rbegin() = '\n';
            // Write row to output stream
            output_file_ << row_str;
        }

    public:
        void flush() {
            output_file_.flush();
        }
    };

    

} // namespace spy