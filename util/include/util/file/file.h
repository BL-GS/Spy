#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>

#include "util/file/file_view.h"
#include "util/file/mapper.h"

namespace spy {

	struct ViewMap {
	public:
        /// Mapping containe <start, end> pair denoting the range of views
		std::map<size_t, size_t> view_map;

	public:
		void add_view(const size_t range_start, const size_t range_end) {
			// If the map is empty, insert directly
			if (view_map.empty()) {
				view_map[range_start] = range_end;
				return;
			}

			auto prev_iter = view_map.lower_bound(range_start);
			auto next_iter = prev_iter--;

			// If all view has larger start address than the new one
			if (next_iter == view_map.begin()) {
				const size_t next_start = next_iter->first;
				// Check whetheer overlap
				if (range_end > next_start) {
					throw std::runtime_error("Expect range pair not to be overlapped");
				}
				// Make fake entity
				auto iter_pair = view_map.insert({range_start, range_start});
				prev_iter = next_iter = iter_pair.first;
				++next_iter;
			}

			const size_t prev_end = prev_iter->second;

			// Check whether overlap
			if (prev_end > range_start) {
				throw std::runtime_error("Expect range pair not to be overlapped");
			}
			if (next_iter != view_map.end()) {
				const size_t next_start = next_iter->first;
				if (range_end > next_start) {
					throw std::runtime_error("Expect range pair not to be overlapped");
				}
			}

			auto cur_iter = prev_iter;

			// Inset or merge with the prev one
			if (prev_end == range_start) {
				prev_iter->second = range_end;
			}
			else {
				auto iter_pair = view_map.insert({range_start, range_end});
				cur_iter = iter_pair.first;
			}

			// Merge with the consequent views if possible
			while (true) {
				prev_iter = cur_iter++;
				if (cur_iter == view_map.end()) { break; }

				const size_t prev_end  = prev_iter->second;
				const size_t cur_start = cur_iter->first;
				const size_t cur_end   = cur_iter->second;

                // Break if they are not adjacent
				if (prev_end != cur_start) { break; }

				// Merget
				prev_iter->second = cur_end;
				cur_iter = view_map.erase(cur_iter);
				// Reset iterator
				--cur_iter;
			}
		}

		void remove_view(const size_t range_start, const size_t range_end) {
			if (view_map.empty()) {
				throw std::runtime_error("Cannot remove view from empty map");
			}

			auto prev_iter = view_map.lower_bound(range_start);
			if (prev_iter == view_map.begin()) {
				throw std::runtime_error("Cannot remove view which hasn't been added");
			}

			--prev_iter;

			const size_t prev_start = prev_iter->first;
			const size_t prev_end   = prev_iter->second;

			if (prev_end < range_end) {
				throw std::runtime_error("Cannot remove view spanning over multiple sub-view");
			}

			if (prev_start == range_start) {
				view_map.erase(prev_iter);
				if (prev_end != range_end) {
					// Remove the fronted part
					add_view(range_end, prev_end);
				}
			}
			else {
				if (prev_end == range_end) {
					// Remove the behind part
					prev_iter->second = range_start;
				}
				else {
					// Remove the middle part
					prev_iter->second = range_start;
					add_view(range_end, prev_end);
				}
			}
		}
	};

} // namespace spy