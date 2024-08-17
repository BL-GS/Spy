#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <functional>
#include <thread>
#include <unordered_map>
#include <stdexcept>
#include <system_error>

#include "loader/file/exception.h"

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
#endif

namespace spy {

    struct FileView {
	public:
		/// The inherent offset of the view, which should be added into that of user
        int64_t view_offset = 0;
		/// The size of the view
        size_t  view_size   = 0;

    public:
		FileView() = default;

        FileView(int64_t offset, size_t size): view_offset(offset), view_size(size) {}

        FileView(const FileView &other) = default;

		FileView(FileView &&other) noexcept = default;

        virtual ~FileView() noexcept = default;

    public:
        virtual int read(void *dst, int64_t offset, size_t size)        = 0;

        virtual int write(const void *src, int64_t offset, size_t size) = 0;
    };

    struct FileSpanView: FileView {
	public:
		/// The start address of the view
        std::byte * view_addr   = nullptr;
		/// The offset of the view to the file
		int64_t     file_offset = 0;

    public:
		FileSpanView() = default;

        FileSpanView(int64_t offset, int64_t size, void *addr): FileView(offset, size), view_addr(static_cast<std::byte *>(addr)) {}

        FileSpanView(const FileSpanView &other) = default;

		FileSpanView(FileSpanView &&other) noexcept = default;

         ~FileSpanView() noexcept override = default;

    public:
        int read(void *dst, int64_t offset, size_t size) override {
	        spy_assert_debug(offset > 0, "out of range read");
	        spy_assert_debug(offset + size <= view_size, "out of range read");
            std::memcpy(dst, view_addr + view_offset + offset, size);
            return -1;
        }

        int write(const void *src, int64_t offset, size_t size) override {
	        spy_assert_debug(offset > 0, "out of range write");
			spy_assert_debug(offset + size <= view_size, "out of range write");
            std::memcpy(view_addr + view_offset + offset, src, size);
            return -1;
        }

        std::byte *deref(int64_t offset) const {
			return view_addr + view_offset + offset;
		}
    };

#ifdef _WIN32

    struct FileMappingView final: public FileSpanView {
	public:
		static constexpr size_t ALIGNED_GRANULARITY = 64 * 1024; // 64K

    public:
        FileMappingView() = default;

        FileMappingView(const HANDLE mapping_handle, int64_t offset, size_t size, bool write = false) {
            // Do not call mmap if size == 0
            if (size == 0) {
                view_offset = 0;
                return;
            }

            const auto prot  = write ? FILE_MAP_WRITE | FILE_MAP_READ : FILE_MAP_READ;

            // Align offset and size by the granularity of OS
            const size_t  aligned_offset = static_cast<size_t>(offset) / ALIGNED_GRANULARITY * ALIGNED_GRANULARITY;
            const size_t  aligned_size   = size + (offset - aligned_offset);

            view_offset = offset - aligned_offset;
            view_size   = aligned_size;
            file_offset = offset;

            // mmap
            {
                constexpr size_t offset_mask    = 0xFFFFFFFF;
                const DWORD offset_high         = (aligned_offset >> 32U) & offset_mask;
                const DWORD offset_low          = aligned_offset & offset_mask;
                view_addr = static_cast<std::byte *>(MapViewOfFile(mapping_handle, prot, offset_high, offset_low, aligned_size));

                if (view_addr == nullptr) {
                    spy_error("failed mmap file (MapViewOfFile): {}", system_error());
                    throw SpyOSFileException("failed MapViewOfFile");
                }
            }
        }

        ~FileMappingView() noexcept override {
            if (view_addr != nullptr) {
                if (UnmapViewOfFile(view_addr) == 0) { spy_warn("failed UnmapViewOfFile"); }
            }
        }

        FileMappingView(FileMappingView&& other) noexcept {
			view_offset      = other.view_offset;
			view_size        = other.view_size;
			view_addr        = other.view_addr;
			file_offset      = other.file_offset;

            other.view_offset      = 0;
            other.view_size        = 0;
            other.view_addr        = nullptr;
            other.file_offset      = 0;
        }

        FileMappingView &operator = (FileMappingView &&other) noexcept {
            view_offset      = other.view_offset;
            view_size        = other.view_size;
            view_addr        = other.view_addr;
            file_offset      = other.file_offset;

            other.view_offset      = 0;
            other.view_size        = 0;
            other.view_addr        = nullptr;
            other.file_offset      = 0;

            return *this;
        }

    public:
        void prefetch(int64_t offset, size_t size) {
            const size_t aligned_offset = offset / ALIGNED_GRANULARITY * ALIGNED_GRANULARITY;
            const size_t aligned_size   = size + (offset - aligned_offset);

            // PrefetchVirtualMemory is only present on Windows 8 and above, so we dynamically load it
            BOOL(WINAPI * pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            // may fail on pre-Windows 8 systems
            pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)> (GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

            if (pPrefetchVirtualMemory != nullptr) {
                // advise the kernel to preload the mapped memory
                WIN32_MEMORY_RANGE_ENTRY range;

                range.VirtualAddress = view_addr + aligned_offset;
                range.NumberOfBytes  = static_cast<SIZE_T>(aligned_size);
                if (pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) == 0) {
                    spy_warn("failed PrefetchVirtualMemory");
                }
            }
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
            SetFilePointer(file_handle_, offset + this->view_offset, nullptr, FILE_BEGIN);
            BOOL ret = ReadFile(file_handle_, dst, size, nullptr, nullptr);
            return ret ? 0 : -1;
        }

        int write(const void *src, int64_t offset, size_t size) override {
            SetFilePointer(file_handle_, offset + this->view_offset, nullptr, FILE_BEGIN);
            BOOL ret = WriteFile(file_handle_, src, size, nullptr, nullptr);
            return ret ? 0 : -1;
        }
    };

    struct ASyncFileView final: public FileView {
    public:
        static constexpr int MAX_RETRY_TIMES = 5;

        struct AsyncTask {
            std::unique_ptr<OVERLAPPED> overlapped;
            std::function<void()>       callback;

            AsyncTask() = default;
            AsyncTask(std::unique_ptr<OVERLAPPED> &&overlapped): overlapped(std::move(overlapped)) {}
            AsyncTask(std::unique_ptr<OVERLAPPED> &&overlapped, std::function<void()> &&callback):
                overlapped(std::move(overlapped)), callback(std::forward<std::function<void()>>(callback)) {}

            AsyncTask& operator =(AsyncTask&& other) = default;
        };
    private:
        /// The handle of the file, which should not be freed by view
        HANDLE file_handle_ = INVALID_HANDLE_VALUE;

        int event_counter_  = 0;
        /// Temporal storage of event handle
        std::unordered_map<int, AsyncTask> event_map_;

    public:
        ASyncFileView() = default;

        ASyncFileView(HANDLE file_handle, int64_t offset, size_t size): FileView(offset, size), file_handle_(file_handle) {}

        ASyncFileView(const ASyncFileView &other) = default;

        ~ASyncFileView() noexcept override {
            sync_all();
        }

    public:
        int read(void *dst, int64_t offset, size_t size) override {
            std::unique_ptr<OVERLAPPED> overlapped_ptr = std::make_unique<OVERLAPPED>();
            std::memset(overlapped_ptr.get(), 0, sizeof(OVERLAPPED));

            offset += this->view_offset;

            constexpr DWORD offset_mask = 0xFFFFFFFF;
            overlapped_ptr->Offset      = offset & offset_mask;
            overlapped_ptr->OffsetHigh  = offset >> 32;
            overlapped_ptr->hEvent      = CreateEvent(nullptr, TRUE, FALSE, nullptr);

            for (int retry_times = 0; retry_times < MAX_RETRY_TIMES; ++retry_times) {
                BOOL ret = ReadFile(file_handle_, dst, size, nullptr, overlapped_ptr.get());
                if (!ret) {
                    DWORD err = GetLastError();
                    switch (err) {
                    case ERROR_IO_PENDING: {
                        const int event_idx   = ++event_counter_;
                        event_map_[event_idx] = { std::move(overlapped_ptr) };
                        return event_idx;     
                    }
    
                    // Recoverable error
                    case ERROR_INVALID_USER_BUFFER: 
                    case ERROR_NOT_ENOUGH_QUOTA:
                    case ERROR_NOT_ENOUGH_MEMORY:
                        std::this_thread::yield();
                        continue;

                    default:
						spy_error("failed reading the file: {}", system_error());
                        return -1;
                    }
                } else {
                    // The content has been loaded or can be read rapidly
                    return 0;
                }
            }
            spy_error("retry overlapped read for too many times({}), try to set less io_thread by `--io-thread <num>`\n", MAX_RETRY_TIMES);
            return -1;
        }

        int read(void* dst, int64_t offset, size_t size, std::function<void()>&& callback) {
            std::unique_ptr<OVERLAPPED> overlapped_ptr = std::make_unique<OVERLAPPED>();
            std::memset(overlapped_ptr.get(), 0, sizeof(OVERLAPPED));

            offset += this->view_offset;

            constexpr DWORD offset_mask = 0xFFFFFFFF;
            overlapped_ptr->Offset      = offset & offset_mask;
            overlapped_ptr->OffsetHigh  = offset >> 32;
            overlapped_ptr->hEvent      = CreateEvent(nullptr, TRUE, FALSE, nullptr);

            for (int retry_times = 0; retry_times < MAX_RETRY_TIMES; ++retry_times) {
                BOOL ret = ReadFile(file_handle_, dst, size, nullptr, overlapped_ptr.get());
                if (!ret) {
                    DWORD err = GetLastError();
                    switch (err) {
                    case ERROR_IO_PENDING: {
                        const int event_idx   = ++event_counter_;
                        event_map_[event_idx] = { std::move(overlapped_ptr), std::forward<std::function<void()>>(callback) };
                        return event_idx;     
                    }    

                    // Recoverable error
                    case ERROR_INVALID_USER_BUFFER: 
                    case ERROR_NOT_ENOUGH_QUOTA:
                    case ERROR_NOT_ENOUGH_MEMORY:
                        std::this_thread::yield();
                        continue;

                    default:
						spy_error("failed reading the file: {}", system_error());
                        return -1;
                    }
                } else {
                    // The content has been loaded or can be read rapidly
                    if (callback) { callback(); }
                    return 0;
                }
            }
            spy_error("retry overlapped read for too many times({}), try to set less io_thread by `--io-thread <num>`\n", MAX_RETRY_TIMES);
            return -1;
        }

        int write(const void *src, int64_t offset, size_t size) override {
            std::unique_ptr<OVERLAPPED> overlapped_ptr = std::make_unique<OVERLAPPED>();
            std::memset(overlapped_ptr.get(), 0, sizeof(OVERLAPPED));

            offset += this->view_offset;

            constexpr DWORD offset_mask = 0xFFFFFFFF;
            overlapped_ptr->Offset       = offset & offset_mask;
            overlapped_ptr->OffsetHigh   = offset >> 32;
            overlapped_ptr->hEvent       = CreateEvent(nullptr, TRUE, FALSE, nullptr);

            for (int retry_times = 0; retry_times < MAX_RETRY_TIMES; ++retry_times) {
                BOOL ret = WriteFile(file_handle_, src, size, nullptr, overlapped_ptr.get());
                if (!ret) {
                    DWORD err = GetLastError();

                    switch (err) {
                    // Reading file
                    case ERROR_IO_PENDING: {
                        const int event_idx   = ++event_counter_;
                        event_map_[event_idx] = { std::move(overlapped_ptr) };
                        return event_idx;
                    }

                    // Recoverable error
                    case ERROR_INVALID_USER_BUFFER: 
                    case ERROR_NOT_ENOUGH_QUOTA:
                    case ERROR_NOT_ENOUGH_MEMORY:
                        std::this_thread::yield();
                        continue;

                    default:
                        spy_error("failed writing the file: {}", system_error());
                        return -1;
                    }
                } else {
                    // The content has been loaded or can be written rapidly
                    return 0;                    
                }
            }
            spy_error("retry overlapped write for too many times({}), try to set less io_thread by `--io-thread <num>`\n", MAX_RETRY_TIMES);
            return -1;
        }

        int write(const void* src, int64_t offset, size_t size, std::function<void()> &&callback) {
            std::unique_ptr<OVERLAPPED> overlapped_ptr = std::make_unique<OVERLAPPED>();
            std::memset(overlapped_ptr.get(), 0, sizeof(OVERLAPPED));

            offset += this->view_offset;

            constexpr DWORD offset_mask = 0xFFFFFFFF;
            overlapped_ptr->Offset       = offset & offset_mask;
            overlapped_ptr->OffsetHigh   = offset >> 32;
            overlapped_ptr->hEvent       = CreateEvent(nullptr, TRUE, FALSE, nullptr);

            for (int retry_times = 0; retry_times < MAX_RETRY_TIMES; ++retry_times) {
                BOOL ret = WriteFile(file_handle_, src, size, nullptr, overlapped_ptr.get());
                if (!ret) {
                    DWORD err = GetLastError();

                    switch (err) {
                    // Reading file
                    case ERROR_IO_PENDING: {
                        const int event_idx   = ++event_counter_;
                        event_map_[event_idx] = { std::move(overlapped_ptr), std::forward<std::function<void()>>(callback) };
                        return event_idx;
                    }

                    // Recoverable error
                    case ERROR_INVALID_USER_BUFFER: 
                    case ERROR_NOT_ENOUGH_QUOTA:
                    case ERROR_NOT_ENOUGH_MEMORY:
                        std::this_thread::yield();
                        continue;

                    default:
						spy_error("failed reading the file: {}", system_error());
                        return -1;
                    }
                } else {
                    // The content has been loaded or can be written rapidly
                    callback();
                    return 0;                    
                }
            }
            spy_error("retry overlapped write for too many times({}), try to set less io_thread by `--io-thread <num>`\n", MAX_RETRY_TIMES);
            return -1;
         }

    public:
        bool sync(int event_idx, DWORD wait_timeout_ms) {
            auto& async_task     = event_map_[event_idx];
            auto& overlapped_ptr = async_task.overlapped;
            auto& callback       = async_task.callback;

            const HANDLE event_handle = overlapped_ptr->hEvent;
            if (event_handle == INVALID_HANDLE_VALUE) { return true; }

            DWORD ret = WaitForSingleObject(event_handle, wait_timeout_ms);
            DWORD transfer_number = 0;

            switch (ret) {
            // success
            case WAIT_OBJECT_0: 
                GetOverlappedResult(file_handle_, overlapped_ptr.get(), &transfer_number, FALSE);

                if (callback) { callback(); }

                CloseHandle(event_handle);
                event_map_.erase(event_idx);
                return true;
            // fail
            case WAIT_TIMEOUT:
                return false;

            case WAIT_ABANDONED:
                spy_warn("waiting for an abandoned event...");
                return false;

            default:
				spy_error("failed waiting for an event: {}", system_error());
                return false; 
            }
        }

        void sync_all() {
            std::vector<int> handle_event_table;
            std::vector<HANDLE> handle_vec;

            handle_event_table.reserve(event_map_.size());
            handle_vec.reserve(event_map_.size());

            for (auto& pair : event_map_) {
                auto& async_task            = pair.second;
                auto& overlapped_ptr        = async_task.overlapped;
                const HANDLE event_handle   = overlapped_ptr->hEvent;

                if (event_handle == INVALID_HANDLE_VALUE) { continue; }
                handle_event_table.emplace_back(pair.first);
                handle_vec.emplace_back(event_handle);
            }
    
            const size_t total_num_events = handle_event_table.size();
            for (size_t num_processed = 0; num_processed < total_num_events; ++num_processed) {
                const DWORD ret = WaitForMultipleObjects(handle_vec.size(), handle_vec.data(), FALSE, INFINITE);
                // there is one finished
                if (ret >= WAIT_OBJECT_0 && ret < WAIT_OBJECT_0 + handle_vec.size()) {
                    const int handle_id  = ret - WAIT_OBJECT_0;
                    const int event_id   = handle_event_table[handle_id];
                    {
                        auto& async_task     = event_map_[event_id];
                        auto& overlapped_ptr = async_task.overlapped;
                        auto& callback       = async_task.callback;

                        DWORD transfer_number = 0;
                        const BOOL success = GetOverlappedResult(file_handle_, overlapped_ptr.get(), &transfer_number, TRUE);                     

                        if (!success) {
                            spy_error("failed get overlapped result: {}", system_error());
                            spy_error("consider using `--io-thread <num>` to decrease the number of concurrent I/O task. (default: 4)");
                            throw SpyOSFileException("failed get overlapped result");
                        } else {
                            if (callback) { callback(); }
                            CloseHandle(overlapped_ptr->hEvent); 
                            overlapped_ptr->hEvent = INVALID_HANDLE_VALUE; 
                        }
                    }
                    handle_event_table.erase(handle_event_table.cbegin() + handle_id);
                    handle_vec.erase(handle_vec.cbegin() + handle_id);
                } else { // fail
                    switch (ret) {

                    case WAIT_ABANDONED:
                        spy_error("wait for overlapped abandoned...");
                        break;

                    default:
                        spy_error("failed waiting for event: {}", system_error());
                        break;
                    }                      
                    throw SpyOSFileException("failed waiting for event");;
                }
            }

            event_map_.clear();
        }

        int sync_one() {
            std::vector<int> handle_event_table;
            std::vector<HANDLE> handle_vec;

            handle_event_table.reserve(event_map_.size());
            handle_vec.reserve(event_map_.size());

            for (auto& pair : event_map_) {
                auto& async_task            = pair.second;
                auto& overlapped_ptr        = async_task.overlapped;
                const HANDLE event_handle   = overlapped_ptr->hEvent;

                if (event_handle == INVALID_HANDLE_VALUE) { continue; }
                handle_event_table.emplace_back(pair.first);
                handle_vec.emplace_back(event_handle);
            }

            const DWORD ret = WaitForMultipleObjects(handle_vec.size(), handle_vec.data(), FALSE, INFINITE);
            // there is one finished
            if (ret >= WAIT_OBJECT_0 && ret < WAIT_OBJECT_0 + handle_vec.size()) {
                const int handle_id  = ret - WAIT_OBJECT_0;
                const int event_id   = handle_event_table[handle_id];
                {
                    auto& async_task     = event_map_[event_id];
                    auto& overlapped_ptr = async_task.overlapped;
                    auto& callback       = async_task.callback;

                    DWORD transfer_number = 0;
                    const BOOL success = GetOverlappedResult(file_handle_, overlapped_ptr.get(), &transfer_number, TRUE);
 
                    if (!success) {
                        spy_error("failed get overlapped result: {}", system_error());
                        spy_error("consider using `--io-thread <num>` to decrease the number of concurrent I/O task. (default: 4)\n");
                        throw SpyOSFileException("failed get overlapped result");
                    } else {
                        if (callback) { callback(); }
                        CloseHandle(overlapped_ptr->hEvent);   
                        event_map_.erase(event_id);
                    }
                }
                return event_id;
            } else { // fail
                switch (ret) {

                case WAIT_ABANDONED:
                    spy_error("wait for overlapped abandoned...");
                    break;

                default:
                    spy_error("failed waiting for event: {}", system_error());
                    break;
                }                      
                throw SpyOSFileException("failed waiting for event");
                return -1;
            }
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

            offset_      = offset - aligned_offset;
            size_        = aligned_size;
            file_offset_ = offset;

			// mmap
			addr_ = mmap(
				nullptr, 
				aligned_size, 
				prot, 
				MAP_SHARED, 
				descriptor, 
				aligned_offset
			);
			if (addr_ == MAP_FAILED) { 
                fprintf(stderr, "failed mmaping file");
                throw std::system_error(errno, std::system_category()); 
            }
		}

		~FileMappingView() noexcept override {
			if (addr_ != nullptr) {
				munmap(addr_, size_);
			}
		}

		FileMappingView(FileMappingView&& other) noexcept : FileSpanView(other) {
			other.reset();
		}

		FileMappingView &operator = (FileMappingView &&other) noexcept {
            offset_      = other.offset_;
            size_        = other.size_;
            file_offset_ = other.file_offset_;
            addr_        = other.addr_;
			other.reset();
			return *this;
		}

	public:
		/*!
		 * @brief Prefetch and allocate necessary memory
		 * @param offset The offset of the range to be prefetched
		 * @param size The size of the range to be prefetched
		 */
		void prefetch(const int64_t offset, const size_t size) {
			madvise(deref(offset), size, MADV_WILLNEED);
		}

	private:
		void reset() {
            offset_      = 0;
            size_        = 0;
            addr_        = nullptr;
            file_offset_ = 0;
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
                fprintf(stderr, "failed to seek file");
				throw std::system_error(errno, std::system_category());
			}
			// Read file synchronously
			::read(descriptor_, dst, size);
			return 0;
		}

		int write(const void *src, int64_t offset, size_t size) override {
			// Move the pointer of file descriptor
			const int64_t cur_offset = lseek(descriptor_, this->offset_ + offset, SEEK_SET);
			if (cur_offset == -1) {
                fprintf(stderr, "failed to seek file");
				throw std::system_error(errno, std::system_category());
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

        struct AsyncTask {
            AIOPointer                  aio_ptr;
            std::function<void()>       callback;

            AsyncTask() = default;
            AsyncTask(AIOPointer &&aio_ptr): aio_ptr(std::move(aio_ptr)) {}
            AsyncTask(AIOPointer &&aio_ptr, std::function<void()> &&callback):
                aio_ptr(std::move(aio_ptr)), callback(std::forward<std::function<void()>>(callback)) {}

            AsyncTask& operator =(AsyncTask&& other) = default;
        };

	private:
		/// The handle of the file, which should not be freed by view
		int descriptor_		= -1;

		int event_counter_  = 0;

		std::unordered_map<int, AsyncTask> event_map_;

	public:
		ASyncFileView() = default;

		ASyncFileView(int descriptor, int64_t offset, size_t size): FileView(offset, size), descriptor_(descriptor) { }

		ASyncFileView(const ASyncFileView &other) = default;

		~ASyncFileView() override { sync_all(); }

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
			if (ret < 0) { 
                fprintf(stderr, "failed reading file asynchronously");
                throw std::system_error(errno, std::system_category()); 
            }

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
			if (ret < 0) { 
                fprintf(stderr, "failed reading file asynchronously");
                throw std::system_error(errno, std::system_category()); 
            }

            // The content has been loaded or can be read rapidly
			const int event_idx 	= ++event_counter_;
			event_map_[event_idx] 	= std::move(aiocb_ptr);
            return event_idx;
		}

        int read(void *dst, int64_t offset, size_t size, std::function<void()> &&callback) {
            AIOPointer aiocb_ptr = std::make_unique<aiocb>();
            std::memset(aiocb_ptr.get(), 0, sizeof(aiocb));

            offset += this->offset_;

			aiocb_ptr->aio_fildes = descriptor_;
			aiocb_ptr->aio_nbytes = size;
			aiocb_ptr->aio_offset = offset;
			aiocb_ptr->aio_buf	  = dst;
			
			int ret = aio_read(aiocb_ptr.get());
			if (ret < 0) { 
                fprintf(stderr, "failed reading file asynchronously");
                throw std::system_error(errno, std::system_category()); 
            }

            // The content has been loaded or can be read rapidly
			const int event_idx 	= ++event_counter_;
			event_map_[event_idx] 	= { std::move(aiocb_ptr), std::forward<std::function<void()>>(callback) };
            return event_idx;
		}

		int write(const void *src, int64_t offset, size_t size, std::function<void()> &&callback) {
            AIOPointer aiocb_ptr = std::make_unique<aiocb>();
            std::memset(aiocb_ptr.get(), 0, sizeof(aiocb));

            offset += this->offset_;

			aiocb_ptr->aio_fildes = descriptor_;
			aiocb_ptr->aio_nbytes = size;
			aiocb_ptr->aio_offset = offset;
			aiocb_ptr->aio_buf	  = const_cast<void *>(src);
			
			int ret = aio_write(aiocb_ptr.get());
			if (ret < 0) { 
                fprintf(stderr, "failed reading file asynchronously");
                throw std::system_error(errno, std::system_category()); 
            }

            // The content has been loaded or can be read rapidly
			const int event_idx 	= ++event_counter_;
			event_map_[event_idx] 	= { std::move(aiocb_ptr), std::forward<std::function<void()>>(callback) };
            return event_idx;
		}

	public:
		bool sync(int event_idx, int64_t wait_timeout_ms) {
            auto &aiocb_pair = event_map_[event_idx];
            auto &aiocb_ptr = aiocb_pair.aio_ptr;
            auto &callback  = aiocb_pair.callback;

			{
				const aiocb *cb_list[1] = { aiocb_ptr.get() };
				timespec wait_time;
                wait_time.tv_sec  = wait_timeout_ms / 1000;
                wait_time.tv_nsec = wait_timeout_ms % 1000 * 1000'000;
				int ret = aio_suspend(cb_list, 1, (wait_timeout_ms == -1) ? nullptr : &wait_time);

				if (ret == -1) {
					if (errno == EAGAIN) { return false; }                          
                    
                    fprintf(stderr, "failed aio sync");
					throw std::system_error(errno, std::system_category());
				}
			}

			const int ret = aio_return(aiocb_ptr.get());
			if (ret == -1) { 
                fprintf(stderr, "failed waiting aio event");
			    throw std::system_error(errno, std::system_category());                
            }

            if (callback) { callback(); }
            return true; 
		}

		void sync_all() {
			for (auto &pair : event_map_) {
                auto &aiocb_pair = pair.second;
                auto &aiocb_ptr = aiocb_pair.aio_ptr;
                auto &callback  = aiocb_pair.callback;
				{
					const aiocb *cb_list[1] = { aiocb_ptr.get() };
					int ret = aio_suspend(cb_list, 1, nullptr);
					if (ret == -1) { 
                        fprintf(stderr, "failed aio sync");
                        throw std::system_error(errno, std::system_category()); 
                    }
				}

				const int ret = aio_return(aiocb_ptr.get());
				if (ret == -1) { 
                    fprintf(stderr, "failed waiting aio event: %s\n", strerror(errno));
                    throw std::system_error(errno, std::system_category()); 
                }
                if (callback) { callback(); }
			}
			event_map_.clear();
		}

		int sync_one() {
			std::vector<int>     handle_event_table;
			std::vector<const aiocb *> cb_vec;

			handle_event_table.reserve(event_map_.size());
			cb_vec.reserve(event_map_.size());

			for (auto& pair : event_map_) {
                auto &aiocb_pair = pair.second;
                auto &aiocb_ptr = aiocb_pair.aio_ptr;

				handle_event_table.push_back(pair.first);
				cb_vec.push_back(aiocb_ptr.get());
			}

			int suspend_ret = aio_suspend(cb_vec.data(), cb_vec.size(), nullptr);
			if (suspend_ret == -1) { 
                fprintf(stderr, "failed aio suspend");
                throw std::system_error(errno, std::system_category()); 
            }

			for (size_t idx = 0; idx < handle_event_table.size(); ++idx) {
				const int event_idx = handle_event_table[idx];
				const aiocb *cb_ptr = cb_vec[idx];

				const int ret = aio_error(cb_ptr);
				if (ret == EINPROGRESS) { continue; }

				if (ret == -1) { 
                    fprintf(stderr, "failed sync one event");
                    throw std::system_error(errno, std::system_category()); 
                }

                auto &aiocb_pair = event_map_[event_idx];
                auto &callback  = aiocb_pair.callback;
                if (callback) { callback(); }

				event_map_.erase(event_idx);

				return event_idx;
			}

			return -1;
		}
	};


#endif // _WIN32 - else

} // namespace spy
