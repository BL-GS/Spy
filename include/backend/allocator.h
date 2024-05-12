/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <array>

#include "util/shell/logger.h"
#include "backend/config.h"

namespace spy {
	template<BackendType T_backend>
	struct BackendProtectedPointer;

	template<BackendType T_backend>
	struct BackendRawPointer;

	/* Use BackendProtectedPointer for memory leak detection */
	template<BackendType T_backend>
	using BackendPointer = BackendRawPointer<T_backend>;


	template<BackendType T_backend>
	class AbstractAllocator {
		friend struct BackendRawPointer<T_backend>;
		friend struct BackendProtectedPointer<T_backend>;

	public:
		AbstractAllocator() = default;
		virtual ~AbstractAllocator() noexcept = default;

	protected:
		virtual void dealloc_mem_inner(void *ptr, size_t size) = 0;

	public:
		virtual BackendPointer<T_backend> alloc_mem(size_t size) = 0;
		virtual void dealloc_mem(BackendPointer<T_backend> &&ptr) = 0;

	};

	template<BackendType T_backend>
	struct BackendProtectedPointer {
	public:
		static constexpr BackendType BACKEND = T_backend;

	private:
		void *ptr_;
		size_t size_;
		AbstractAllocator<T_backend> *allocator_ptr_;

	public:
		BackendProtectedPointer() : ptr_(nullptr), size_(0), allocator_ptr_(nullptr) {}
		BackendProtectedPointer(void *ptr, size_t size, AbstractAllocator<T_backend> *allocator_ptr) : 
			ptr_(ptr), size_(size), allocator_ptr_(allocator_ptr)  {}

	public:
		BackendProtectedPointer(BackendProtectedPointer &&other)  noexcept : 
			ptr_(other.ptr_), size_(other.size_), allocator_ptr_(other.allocator_ptr_) { other.ptr_ = nullptr; }
		~BackendProtectedPointer() {
			if (ptr_ != nullptr) { allocator_ptr_->dealloc_mem_inner(ptr_, size_); }
		}
			
	public:
		void *deref() const { return ptr_;  }
		size_t size() const { return size_;  }
	};

	template<BackendType T_backend>
	struct BackendRawPointer {
	public:
		static constexpr BackendType BACKEND = T_backend;

	private:
		void *ptr_;
		size_t size_;

	public:
		BackendRawPointer() : ptr_(nullptr), size_(0) {}
		BackendRawPointer(void *ptr, size_t size, [[maybe_unused]] AbstractAllocator<T_backend> *allocator_ptr) :
			ptr_(ptr), size_(size) {}

	public:
		BackendRawPointer(BackendRawPointer &&other) : ptr_(other.ptr_), size_(other.size_) { other.ptr_ = nullptr; }
		~BackendRawPointer() = default;

	public:
		void *deref() const { return ptr_; }
		size_t size() const { return size_; }
	};


	template<BackendType T_backend>
	class SimpleAllocator : public AbstractAllocator<T_backend> {
	private:
		struct Page {
			Page *next_ptr;
		};

	public:
		static constexpr size_t MIN_ALLOC_SIZE_LOG = 4;
		static constexpr size_t MIN_ALLOC_SIZE = 1 << MIN_ALLOC_SIZE_LOG;
		static constexpr size_t MAX_ALLOC_SIZE_LOG = 12;
		static constexpr size_t MAX_ALLOC_SIZE = 1 << MAX_ALLOC_SIZE_LOG;

		static constexpr size_t NUM_BUCKET = MAX_ALLOC_SIZE_LOG - MIN_ALLOC_SIZE_LOG + 1;
		static constexpr size_t INIT_PAGE_NUM = 8;

		static_assert(MIN_ALLOC_SIZE > sizeof(void *));

	private:
		AbstractBackend *backend_ptr_;
		size_t total_size_;
		std::array<Page *, NUM_BUCKET> page_bucket_;

	public:
		SimpleAllocator(AbstractBackend *backend_ptr) : backend_ptr_(backend_ptr), total_size_(0) {
			page_bucket_.fill(nullptr);

			for (size_t bucket_idx = 0; bucket_idx < NUM_BUCKET; ++bucket_idx) {
				const size_t page_size = (1 << (bucket_idx + MIN_ALLOC_SIZE_LOG));
				reserve_pages(bucket_idx, INIT_PAGE_NUM);
				total_size_ += page_size * INIT_PAGE_NUM;
			}
		}

		~SimpleAllocator() noexcept override {
			size_t total_dealloc_size = 0;
			for (size_t bucket_idx = 0; bucket_idx < NUM_BUCKET; ++bucket_idx) {
				const size_t page_size = page_idx_to_size(bucket_idx);
				Page *cur_page_ptr = page_bucket_[bucket_idx];

				while (cur_page_ptr != nullptr) {
					Page *next_page_ptr = cur_page_ptr->next_ptr;
					backend_ptr_->dealloc_memory(cur_page_ptr, page_size);
					cur_page_ptr = next_page_ptr;

					total_dealloc_size += page_size;
				}
			}

			if (total_dealloc_size != total_size_) {
				spy_warn("The size of page deallocated is different from that allocated (alloc: {}, dealloc: {})", total_size_, total_dealloc_size);
			}
		}

	public:
		BackendPointer<T_backend> alloc_mem(const size_t size) override {
			if (size > MAX_ALLOC_SIZE) {
				void *res_ptr = backend_ptr_->alloc_memory(size);
				return {res_ptr, size, this};
			}
			const size_t bucket_idx = size_to_page_idx(size);
			const size_t actual_size = (1 << (bucket_idx + MIN_ALLOC_SIZE_LOG));

			Page *cur_page_ptr = page_bucket_[bucket_idx];
			if (cur_page_ptr == nullptr) {
				reserve_pages(bucket_idx, INIT_PAGE_NUM);
				cur_page_ptr = page_bucket_[bucket_idx];
			}

			Page *next_page_ptr = cur_page_ptr->next_ptr;
			page_bucket_[bucket_idx] = next_page_ptr;
			return {cur_page_ptr, actual_size, this};
		}

		void dealloc_mem(BackendPointer<T_backend> &&ptr) override {
			Page *page_ptr = static_cast<Page *>(ptr.deref());
			const size_t actual_size = ptr.size();

			if (actual_size > MAX_ALLOC_SIZE) {
				backend_ptr_->dealloc_memory(page_ptr, actual_size);
			} else {
				const size_t bucket_idx = size_to_page_idx(actual_size);
				const size_t page_size = page_idx_to_size(bucket_idx);
				spy_assert(page_size == actual_size, "The size of page should be equal to the actual size (expect: {}, given: {})", page_size, actual_size);

				page_ptr->next_ptr = page_bucket_[bucket_idx];
				page_bucket_[bucket_idx] = page_ptr;
			}
		}

	protected:
		void dealloc_mem_inner(void *ptr, const size_t size) override {
			dealloc_mem({ptr, size, this});
		}

		void reserve_pages(const size_t bucket_idx, const size_t page_num) {
			const size_t page_size = page_idx_to_size(bucket_idx);
			auto &cur_bucket = page_bucket_[bucket_idx];
			Page *prev_page_ptr = nullptr;

			for (size_t page_idx = 0; page_idx < page_num; ++page_idx) {
				Page *new_page_ptr = static_cast<Page *>(backend_ptr_->alloc_memory(page_size));
				new_page_ptr->next_ptr = prev_page_ptr;
				prev_page_ptr = new_page_ptr;
			}

			cur_bucket = prev_page_ptr;
		}

	private:
		static constexpr size_t page_idx_to_size(const size_t idx) {
			return (1 << (idx + MIN_ALLOC_SIZE_LOG)); 
		}

		static constexpr size_t size_to_page_idx(const size_t size) { 
			size_t idx = MIN_ALLOC_SIZE_LOG;
			while ((1 << idx) < size) { ++idx; }
			return idx - MIN_ALLOC_SIZE_LOG;
		}
	};

}