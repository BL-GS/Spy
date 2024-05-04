#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <memory>
#include <thread>

#ifdef __has_include
	#if __has_include(<unistd.h>)
		#include <unistd.h>
		#if defined(_POSIX_MAPPED_FILES)
			#include <sys/mman.h>
			#include <fcntl.h>
		#endif
		#if defined(_POSIX_MEMLOCK_RANGE)
			#include <sys/resource.h>
		#endif
	#endif
#endif

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
	#include <io.h>
#else
	#include <aio.h>
	#include "async/loop.h"
#endif

#include "util/logger.h"
#include "util/file/type.h"

namespace spy {

	/*!
	 * @brief The definition view of a range of the file
	 * @detail This structure denotes a 1D view of the file.
	 * | --- offset --- | --- size --- | ---...
	 * | ---------------| /// view /// | ---...
	 * ------------------==============---------
	 */
	struct FileView {
	protected:
		/// The offset of the view on file
		int64_t offset_;
		/// The size of the view on file
		size_t size_;

	public:
		FileView(): offset_(0), size_(0) {}

		FileView(int64_t offset, size_t size): offset_(offset), size_(size) {}

		FileView(const FileView &other) = default;

		virtual ~FileView() noexcept = default;

		FileView &operator=(const FileView &other) = default;

		bool operator==(const FileView &other) const { return offset_ == other.offset_ && size_ == other.size_; }

	public:
		/*!
		 * @brief Read from view the a specific buffer
		 * @param dst The location where data is to write
		 * @param offset The offset of data on the view
		 * @param size The size of data to read
		 * @return 
		 * For asynchronous operation, this function returns 0 if finished immediately, 
		 *  returns negatives if failed, or return the id of task which can be used for synchronization.
		 * For synchronous operation, this function return 0 if finished successfully, or negatives if failed. 
		 */
		virtual int read(void *dst, int64_t offset, size_t size)        = 0;

		/*!
		 * @brief Write specific data from buffer to the view
		 * @param dst The location where the source data from
		 * @param offset The offset of buffer on the view which is to written
		 * @param size The size of data to write
		 * @return 
		 * For asynchronous operation, this function returns 0 if finished immediately, 
		 *  returns negatives if failed, or return the id of task which can be used for synchronization.
		 * For synchronous operation, this function return 0 if finished successfully, or negatives if failed. 
		 */
		virtual int write(const void *src, int64_t offset, size_t size) = 0;

	 public:
		/*!
		 * @brief Get the offset of the view on file
		 */
		size_t offset() const { return offset_; }

		/*!
		 * @brief Get the size of the view
		 */
		size_t   size() const { return size_;   }
	};

	/*!
	 * @brief For addressable device, this structure gives an simple definition of operations on the view.
	 * On this view, data are read/written by virtual address directly.
	 */
	struct FileSpanView: FileView {
	protected:
		/// The start address of the view (which should has been offseted by offset_)
		void *addr_;

	public:
		FileSpanView(): addr_(nullptr) {}

		FileSpanView(int64_t offset, size_t size, void *addr): FileView(offset, size), addr_(addr) {}

		FileSpanView(const FileSpanView &other) = default;

		 ~FileSpanView() noexcept override = default;

	public:
		 /*!
		 * @brief Read from view the a specific buffer
		 * @param dst The location where data is to write
		 * @param offset The offset of data on the view
		 * @param size The size of data to read
		 * @return 0 on success, -1 on failure
		 * @note The given offset should not be confused with the offset_, which was set by the constructor. 
		 * In other words, the given offset here should not contain the offset of the view.
		 */
		int read(void *dst, int64_t offset, size_t size) override {
			std::memcpy(dst, static_cast<uint8_t *>(addr_) + offset, size);
			return -1;
		}

		/*!
		 * @brief Write specific data from buffer to the view
		 * @param dst The location where the source data from
		 * @param offset The offset of buffer on the view which is to written
		 * @param size The size of data to write
		 * @return 0 on success, -1 on failure
		 * @note The given offset should not be confused with the offset_, which was set by the constructor. 
		 * In other words, the given offset here should not contain the offset of the view.
		 */
		int write(const void *src, int64_t offset, size_t size) override {
			std::memcpy(static_cast<uint8_t *>(addr_) + offset, src, size);
			return -1;
		}

	public:
		/// Get the start address of the view
		void *deref()               const { return addr_; }

		/// Get the address offsetted of the view
		void *deref(size_t offset)  const { return static_cast<uint8_t *>(addr_) + offset_ + offset; }
	};

#ifdef _WIN32
	
	struct FileMappingView final: public FileSpanView {
	public:
		static constexpr int ALIGNED_GRANULARITY = 64 * 1024; // 64K

	public:
		FileMappingView() = default;

		FileMappingView(const HANDLE mapping_handle, int64_t offset, size_t size, bool write = false) {
			// Do not call mmap if size == 0
			if (size == 0) {
				offset_ = 0;
				return;
			}

			const auto prot  = write ? FILE_MAP_WRITE | FILE_MAP_READ : FILE_MAP_READ;

			// Align offset and size by the granularity of OS
			const int64_t aligned_offset = offset / ALIGNED_GRANULARITY * ALIGNED_GRANULARITY;
			const size_t  aligned_size   = size + (offset - aligned_offset);

			offset_ = offset - aligned_offset;
			size_   = aligned_size;

			// mmap
			const LARGE_INTEGER mapping_offset { .QuadPart = aligned_offset };
			addr_ = static_cast<uint8_t*>(MapViewOfFile(mapping_handle, 
				prot, 
				mapping_offset.HighPart, 
				mapping_offset.LowPart, 
				aligned_size
			));
			if (addr_ == nullptr) { throw SpyOSFileException("failed mapping file"); }
		}

		~FileMappingView() noexcept override {
			if (addr_ != nullptr) {
				const auto ret = UnmapViewOfFile(addr_);
				SPY_ASSERT_NOEXCEPTION(ret != 0, "failed unmapping file");
			}
		}

		FileMappingView(FileMappingView&& other) noexcept : FileSpanView(other) {
			other.reset();
		}

		FileMappingView &operator = (FileMappingView &&other) noexcept {
			addr_   = other.addr_;
			offset_ = other.offset_;
			size_   = other.size_;
			other.reset();
			return *this;
		}

	public:
		/*!
		 * @brief Prefetch and allocate necessary memory
		 * @param offset The offset of the range to be prefetched
		 * @param size The size of the range to be prefetched
		 */
		void prefetch(const size_t offset, const size_t size) {
			// PrefetchVirtualMemory is only present on Windows 8 and above, so we dynamically load it
			BOOL(WINAPI * pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
			HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

			// may fail on pre-Windows 8 systems
			pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)> (GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

			if (pPrefetchVirtualMemory != nullptr) {
				// advise the kernel to preload the mapped memory
				WIN32_MEMORY_RANGE_ENTRY range{
					.VirtualAddress = static_cast<uint8_t *>(addr_) + offset_,
					.NumberOfBytes  = static_cast<SIZE_T>(size_)
				};

				if (pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) == 0) {
					SPY_WARN("failed prefetching virtual memory");
				}
			}
		}

	private:
		void reset() {
			addr_   = nullptr;
			offset_ = 0;
			size_   = 0;
		}
	};

	struct SyncFileView final: public FileView {
	private:
		/// The handle of the file, which should not be freed by view
		HANDLE file_handle_;

	public:
		SyncFileView() = default;

		SyncFileView(HANDLE file_handle, int64_t offset, size_t size): FileView(offset, size), file_handle_(file_handle) {}

		SyncFileView(const SyncFileView &other) = default;

		~SyncFileView() noexcept override       = default;

	public:
		int read(void *dst, int64_t offset, size_t size) override {
			// Move the pointer of file descriptor
			LARGE_INTEGER offset_integer { .QuadPart = this->offset_ + offset };
			offset_integer.LowPart = SetFilePointer(file_handle_, offset_integer.LowPart, &offset_integer.HighPart, FILE_BEGIN);
			if (offset_integer.LowPart == INVALID_SET_FILE_POINTER) {
				throw SpyOSFileException("failed to seek file");
			}
			// Read file synchronously
			ReadFile(file_handle_, dst, size, nullptr, nullptr);
			return 0;
		}

		int write(const void *src, int64_t offset, size_t size) override {
			// Move the pointer of file descriptor
			LARGE_INTEGER offset_integer { .QuadPart = this->offset_ + offset };
			offset_integer.LowPart = SetFilePointer(file_handle_, offset_integer.LowPart, &offset_integer.HighPart, FILE_BEGIN);
			if (offset_integer.LowPart == INVALID_SET_FILE_POINTER) {
				throw SpyOSFileException("failed to seek file");
			}
			// Write file synchronously
			WriteFile(file_handle_, src, size, nullptr, nullptr);
			return 0;
		}
	};

	struct ASyncFileView final: public FileView {
	private:
		using OverlappedPointer = std::unique_ptr<OVERLAPPED>;
	private:
		/// The handle of the file, which should not be freed by view
		HANDLE file_handle_ = INVALID_HANDLE_VALUE;

		int event_counter_  = 0;
		/// Temporal storage of event handle
		std::unordered_map<int, OverlappedPointer> event_map_;

	public:
		ASyncFileView() = default;

		ASyncFileView(HANDLE file_handle, int64_t offset, size_t size): FileView(offset, size), file_handle_(file_handle) {}

		ASyncFileView(const ASyncFileView &other) = default;

		~ASyncFileView() noexcept override {
			for (int i = 1; i <= event_counter_; ++i) {
				constexpr auto timeout_ms = 5000;
				bool success = sync<false>(i, timeout_ms);
				SPY_ASSERT_NOEXCEPTION(success, "Failed sync with event");
			}
		}

	public:
		int read(void *dst, int64_t offset, size_t size) override {
            OverlappedPointer overlapped_ptr = std::make_unique<OVERLAPPED>();
            std::memset(overlapped_ptr.get(), 0, sizeof(OVERLAPPED));

            offset += this->offset_;

            constexpr DWORD offset_mask = 0xFFFFFFFF;
            overlapped_ptr->Offset 		= offset & offset_mask;
            overlapped_ptr->OffsetHigh 	= offset >> 32;
            overlapped_ptr->hEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);

            BOOL ret = ReadFile(file_handle_, dst, size, nullptr, overlapped_ptr.get());
            if (ret == FALSE) {
                DWORD err = GetLastError();
                if (err == ERROR_IO_PENDING) { 
                    const int event_idx = ++event_counter_;
                    event_map_[event_idx] = std::move(overlapped_ptr);
                    return event_idx; 
                }
                throw SpyOSFileException("failed reading file");
            }

            // The content has been loaded or can be read rapidlly
            return 0;
		}

		int write(const void *src, int64_t offset, size_t size) override {
            OverlappedPointer overlapped_ptr = std::make_unique<OVERLAPPED>();
            std::memset(overlapped_ptr.get(), 0, sizeof(OVERLAPPED));

            offset += this->offset_;

            constexpr DWORD offset_mask = 0xFFFFFFFF;
            overlapped_ptr->Offset       = offset & offset_mask;
            overlapped_ptr->OffsetHigh   = offset >> 32;
            overlapped_ptr->hEvent       = CreateEvent(nullptr, TRUE, FALSE, nullptr);

            BOOL ret = WriteFile(file_handle_, src, size, nullptr, overlapped_ptr.get());
            while (ret == FALSE) {
                const DWORD err = GetLastError();

                switch (err) {
                // Reading file
                case ERROR_IO_PENDING: {
                    int event_idx = ++event_counter_;
                    event_map_[event_idx] = std::move(overlapped_ptr);
                    return event_idx;
                }

                // Recoverable error
                case ERROR_INVALID_USER_BUFFER: 
                case ERROR_NOT_ENOUGH_QUOTA:
                case ERROR_NOT_ENOUGH_MEMORY:
                    std::this_thread::yield();
                    continue;

                default:
                    throw SpyOSFileException("failed reading file");
                }
            }

            // The content has been loaded or can be write rapidlly
            return 0;
		}

	public:
		bool poll(int event_idx) {
            auto &overlapped_ptr      = event_map_[event_idx];
            const HANDLE event_handle   = overlapped_ptr->hEvent;
            if (event_handle == INVALID_HANDLE_VALUE) { return true; }

            DWORD transfer_number = 0;
            const BOOL ret = GetOverlappedResult(file_handle_, overlapped_ptr.get(), &transfer_number, FALSE);
            if (ret == TRUE) { return true; }

            const DWORD err = GetLastError();
            if (err != ERROR_IO_INCOMPLETE) { 
                throw SpyOSFileException("failed polling event");
            }
            return false; 
		}

		template<bool T_exception = true>
		bool sync(int event_idx, DWORD wait_timeout_ms) {
            auto& overlapped_ptr = event_map_[event_idx];
            const HANDLE event_handle = overlapped_ptr->hEvent;
            if (event_handle == INVALID_HANDLE_VALUE) { return true; }

            const DWORD ret = WaitForSingleObject(event_handle, wait_timeout_ms);
            DWORD transfer_number = 0;

            switch (ret) {
            // success
            case WAIT_OBJECT_0: 
                GetOverlappedResult(file_handle_, overlapped_ptr.get(), &transfer_number, FALSE);
                CloseHandle(event_handle);
                event_map_.erase(event_idx);
                return true;
            // fail
            case WAIT_TIMEOUT:
                return false;

            case WAIT_ABANDONED:
                SPY_WARN("wait for overlapped abandoned...");
                return false;

            default:
				if constexpr (T_exception) {
					throw SpyOSFileException("failed waiting for event");
				}
                return false; 
            }
		}

		void sync_all() {
			for (auto &pair : event_map_) {
				auto& overlapped_ptr = pair.second;
				const HANDLE event_handle = overlapped_ptr->hEvent;
				if (event_handle == INVALID_HANDLE_VALUE) { continue; }

				DWORD transfer_number = 0;
				const BOOL success = GetOverlappedResult(file_handle_, 
					overlapped_ptr.get(), 
					&transfer_number, 
					TRUE
				);
				overlapped_ptr->hEvent = INVALID_HANDLE_VALUE;
			
				if (success == FALSE) { throw SpyOSFileException("failed waiting for event"); }
			}
			event_map_.clear();
		}

		int sync_one() {
			std::vector<int> handle_event_table;
			std::vector<HANDLE> handle_vec;

			handle_event_table.reserve(event_map_.size());
			handle_vec.reserve(event_map_.size());

			for (auto& pair : event_map_) {
				auto& overlapped_ptr = pair.second;
				const HANDLE event_handle = overlapped_ptr->hEvent;

				if (event_handle == INVALID_HANDLE_VALUE) { continue; }
				handle_event_table.push_back(pair.first);
				handle_vec.push_back(event_handle);
			}

			const DWORD ret = WaitForMultipleObjects(
				handle_vec.size(), 
				handle_vec.data(), 
				FALSE, 
				INFINITE
			);
			DWORD transfer_number = 0;
			if (ret >= WAIT_OBJECT_0 && ret < WAIT_OBJECT_0 + handle_vec.size()) {
				// there is one finished
				const int event_id = ret - WAIT_OBJECT_0;
				auto& overlapped_ptr = event_map_[event_id];

				GetOverlappedResult(file_handle_, overlapped_ptr.get(), &transfer_number, TRUE);
				CloseHandle(overlapped_ptr->hEvent);
				event_map_.erase(event_id);

				return handle_event_table[event_id];
			}

			switch (ret) {
				// fail
			case WAIT_TIMEOUT:
				break;

			case WAIT_ABANDONED:
				SPY_WARN("wait for overlapped abandoned...");
				break;

			default:
				throw SpyOSFileException("failed waiting for event");
			}
			return -1;
		}
	};

#else // _WIN32

	struct FileMappingView final: public FileSpanView {
	public:
		static constexpr int ALIGNED_GRANULARITY = 4 * 1024; // 4K

	public:
		FileMappingView() = default;

		FileMappingView(const int descriptor, int64_t offset, size_t size, bool write = false) {
			// Do not call mmap if size == 0
			if (size == 0) {
				offset_ = 0;
				return;
			}

			const int prot = write ? PROT_READ | PROT_WRITE : PROT_READ;

			// Align offset and size by the granularity of OS
			const int64_t aligned_offset = offset / ALIGNED_GRANULARITY * ALIGNED_GRANULARITY;
			const size_t  aligned_size   = size + (offset - aligned_offset);

			offset_ = offset - aligned_offset;
			size_   = aligned_size;

			// mmap
			addr_ = mmap(
				nullptr, 
				aligned_size, 
				prot, 
				MAP_SHARED, 
				descriptor, 
				aligned_offset
			);
			if (addr_ == MAP_FAILED) { throw SpyOSFileException("failed mapping file"); }
		}

		~FileMappingView() noexcept override {
			if (addr_ != nullptr) {
				const auto ret = munmap(addr_, size_);
				SPY_ASSERT_NOEXCEPTION(ret == 0, "failed unmapping file");
			}
		}

		FileMappingView(FileMappingView&& other) noexcept : FileSpanView(other) {
			other.reset();
		}

		FileMappingView &operator = (FileMappingView &&other) noexcept {
			addr_   = other.addr_;
			offset_ = other.offset_;
			size_   = other.size_;
			other.reset();
			return *this;
		}

	public:
		/*!
		 * @brief Prefetch and allocate necessary memory
		 * @param offset The offset of the range to be prefetched
		 * @param size The size of the range to be prefetched
		 */
		void prefetch(const size_t offset, const size_t size) {
			madvise(deref(offset), size, MADV_WILLNEED);
		}

	private:
		void reset() {
			addr_   = nullptr;
			offset_ = 0;
			size_   = 0;
		}
	};

	struct SyncFileView final: public FileView {
	private:
		/// The handle of the file, which should not be freed by view
		int descriptor_ = -1;

	public:
		SyncFileView() = default;

		SyncFileView(int descriptor, int64_t offset, size_t size): FileView(offset, size), descriptor_(descriptor) {}

		SyncFileView(const SyncFileView &other) = default;

		~SyncFileView() noexcept override       = default;

	public:
		int read(void *dst, int64_t offset, size_t size) override {
			// Move the pointer of file descriptor
			const int64_t cur_offset = lseek(descriptor_, this->offset_ + offset, SEEK_SET);
			if (cur_offset == -1) {
				throw SpyOSFileException("failed to seek file");
			}
			// Read file synchronously
			::read(descriptor_, dst, size);
			return 0;
		}

		int write(const void *src, int64_t offset, size_t size) override {
			// Move the pointer of file descriptor
			const int64_t cur_offset = lseek(descriptor_, this->offset_ + offset, SEEK_SET);
			if (cur_offset == -1) {
				throw SpyOSFileException("failed to seek file");
			}
			// Read file synchronously
			::write(descriptor_, src, size);
			return 0;
		}
	};

	struct ASyncFileView final: public FileView {
	public:
		using AIOPointer = std::unique_ptr<aiocb>;

		static constexpr size_t IO_URING_QUEUE_LENGTH 	= 8;

		static constexpr size_t BUFFER_UNIT_SIZE 		= 4096;

	private:
		/// The handle of the file, which should not be freed by view
		int descriptor_		= -1;

		int event_counter_  = 0;

		std::unordered_map<int, AIOPointer> event_map_;

	public:
		ASyncFileView() = default;

		ASyncFileView(int descriptor, int64_t offset, size_t size): FileView(offset, size), descriptor_(descriptor) { }

		ASyncFileView(const ASyncFileView &other) = default;

		~ASyncFileView() override {
			sync_all();
		}

	public:
		int read(void *dst, int64_t offset, size_t size) override {
            AIOPointer aiocb_ptr = std::make_unique<aiocb>();
            std::memset(aiocb_ptr.get(), 0, sizeof(aiocb));

            offset += this->offset_;

			aiocb_ptr->aio_fildes = descriptor_;
			aiocb_ptr->aio_nbytes = size;
			aiocb_ptr->aio_offset = offset;
			aiocb_ptr->aio_buf	  = dst;
			
			int ret = aio_read(aiocb_ptr.get());
			if (ret < 0) { throw SpyOSFileException("failed reading file asynchronously"); }

            // The content has been loaded or can be read rapidly
			const int event_idx 	= ++event_counter_;
			event_map_[event_idx] 	= std::move(aiocb_ptr);
            return event_idx;
		}

		int write(const void *src, int64_t offset, size_t size) override {
            AIOPointer aiocb_ptr = std::make_unique<aiocb>();
            std::memset(aiocb_ptr.get(), 0, sizeof(aiocb));

            offset += this->offset_;

			aiocb_ptr->aio_fildes = descriptor_;
			aiocb_ptr->aio_nbytes = size;
			aiocb_ptr->aio_offset = offset;
			aiocb_ptr->aio_buf	  = const_cast<void *>(src);
			
			int ret = aio_write(aiocb_ptr.get());
			if (ret < 0) { throw SpyOSFileException("failed reading file asynchronously"); }

            // The content has been loaded or can be read rapidly
			const int event_idx 	= ++event_counter_;
			event_map_[event_idx] 	= std::move(aiocb_ptr);
            return event_idx;
		}

	public:
		bool poll(int event_idx) {
			auto &aiocb_ptr = event_map_[event_idx];
			{ // poll
				const int ret = aio_error(aiocb_ptr.get());
				// The read/write is under progressing
				if (ret == EINPROGRESS) { return false; }				
			}
			return true;
		}

		template<bool T_exception = true>
		bool sync(int event_idx, int64_t wait_timeout_ms) {
            auto &aiocb_ptr = event_map_[event_idx];

			{
				const aiocb *cb_list[1] = { aiocb_ptr.get() };
				const timespec wait_time{
					.tv_sec  = wait_timeout_ms / 1000,
					.tv_nsec = wait_timeout_ms % 1000 * 1000'000
				};
				int ret = aio_suspend(cb_list, 1, (wait_timeout_ms == -1) ? nullptr : &wait_time);

				if (ret == -1) {
					if (errno == EAGAIN) {
						return false;
					} else {
						throw SpyOSFileException("failed aio sync");
					}
				}
			}

			const int ret = aio_return(aiocb_ptr.get());
			if (ret == 0) { return true; }

			throw SpyOSFileException("failed waiting aio event");
		}

		void sync_all() {
			for (auto &pair : event_map_) {
				auto& aiocb_ptr = pair.second;
				{
					const aiocb *cb_list[1] = { aiocb_ptr.get() };
					int ret = aio_suspend(cb_list, 1, nullptr);
					if (ret == -1) { throw SpyOSFileException("failed aio sync"); }
				}

				const int ret = aio_return(aiocb_ptr.get());
				if (ret != 0) { throw SpyOSFileException("failed waiting aio event"); }
			}
			event_map_.clear();
		}

		int sync_one() {
			std::vector<int>     handle_event_table;
			std::vector<const aiocb *> cb_vec;

			handle_event_table.reserve(event_map_.size());
			cb_vec.reserve(event_map_.size());

			for (auto& pair : event_map_) {
				auto& aiocb_ptr = pair.second;
				handle_event_table.push_back(pair.first);
				cb_vec.push_back(aiocb_ptr.get());
			}

			int suspend_ret = aio_suspend(cb_vec.data(), cb_vec.size(), nullptr);
			if (suspend_ret == -1) { throw SpyOSFileException("failed aio suspend"); }

			for (size_t idx = 0; idx < handle_event_table.size(); ++idx) {
				const int event_idx = handle_event_table[idx];
				const aiocb *cb_ptr = cb_vec[idx];

				const int ret = aio_error(cb_ptr);
				if (ret == EINPROGRESS) { continue; }

				event_map_.erase(event_idx);
				if (ret == -1) { throw SpyOSFileException("failed sync one event"); }
				return event_idx;
			}

			return -1;
		}
	};


#endif // _WIN32 - else

} // namespace spy