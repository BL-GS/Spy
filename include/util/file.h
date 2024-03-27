#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <map>

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
#endif

#include "util/align.h"
#include "util/logger.h"

namespace spy {

#ifdef _WIN32
    constexpr size_t ALIGNED_GRANULARITY = 64 * 1024; // 64K

    inline static size_t windows_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return static_cast<size_t>(si.dwPageSize);
    }

#endif

    struct SpyFile {
    private:
        FILE * fp_;
        size_t size_;

#ifdef _WIN32
        HANDLE fd_;

    public:
        SpyFile(const std::string_view filename, const char *mode = "rb"): fp_(nullptr), fd_(INVALID_HANDLE_VALUE) {
            fp_ = std::fopen(filename.data(), mode);
            SPY_ASSERT_FMT(fp_ != nullptr, "Cannot open file: {}", filename);
            fd_ = (HANDLE)_get_osfhandle(_fileno(fp_));

            seek(0, SEEK_END);
            size_ = tell();
            seek(0, SEEK_SET);
        }

        ~SpyFile() noexcept {
            if (fp_ != nullptr) { std::fclose(fp_); }

            fp_ = nullptr;
            fd_ = INVALID_HANDLE_VALUE;
        } 

        SpyFile(SpyFile &&other): fp_(other.fp_), fd_(other.fd_) {
            other.fp_ = nullptr;
            other.fd_ = INVALID_HANDLE_VALUE;
        }

#else
        int   fd_;

    public:
        SpyFile(const std::string_view filename, const char *mode = "rb"): fp_(nullptr), fd_(-1) {
            fp_ = std::fopen(filename.data(), mode);
            SPY_ASSERT_FMT(fp_ != nullptr, "Cannot open file: {}", filename);
            fd_ = fileno(fp_);
            
            seek(0, SEEK_END);
            size_ = tell();
            seek(0, SEEK_SET);
        }

        ~SpyFile() noexcept {
            if (fp_ != nullptr) { std::fclose(fp_); }

            fp_ = nullptr;
            fd_ = -1;
        }

        SpyFile(SpyFile &&other): fp_(other.fp_), fd_(other.fd_) {
            other.fp_ = nullptr;
            other.fd_ = -1;
        }
#endif


    public:
        auto get_fp() const { return fp_; }

        auto get_fd() const { return fd_; }

        size_t tell() const {
#ifdef _WIN32
            return _ftelli64(fp_);
#else
            return std::ftell(fp_);
#endif
        }

        void seek(size_t offset, int whence) const {
#ifdef _WIN32
            int ret = _fseeki64(fp_, static_cast<__int64>(offset), whence);
#else
            int ret = std::fseek(fp_, (long) offset, whence);
#endif
            SPY_ASSERT(ret == 0, "Failed seek file");
        }

        size_t size() const { return size_; }

    };

    /*
     * To save memory of tensors which has been offloaded to GPU, and avoid dual memory occupation when loading model.
     */

    struct FileView {
    protected:
        size_t offset_;
        size_t size_;

    public:
        FileView(): offset_(0), size_(0) {}
        FileView(size_t offset, size_t size): offset_(offset), size_(size) {}
        FileView(const FileView &other) = default;
        virtual ~FileView() noexcept = default;

    public:
        virtual void read(void *dst, size_t offset, size_t size)        = 0;
        virtual void write(const void *src, size_t offset, size_t size) = 0;

        virtual uint8_t* deref() const = 0;
        virtual uint8_t* deref(size_t offset) const = 0;

        size_t offset() const { return offset_; }
        size_t   size() const { return size_;   }
    };


    struct FileMappingView: public FileView {
    protected:
        uint8_t *addr_;

    public:
#ifdef __linux__
        FileMappingView(const int fd, size_t offset, size_t size, bool write = false): FileView(offset, size), addr_(nullptr) {
            // Do not call mmap if size == 0
            if (size == 0) {
                offset_ = 0;
                return;
            }

            const int prot = write ? PROT_WRITE | PROT_READ : PROT_READ;

            const size_t aligned_offset = offset / 4096 * 4096;
            const size_t aligned_size   = size + (offset - aligned_offset); 

            offset_ = aligned_offset;
            size_   = aligned_size;

            addr_ = static_cast<uint8_t *>(mmap(NULL, aligned_size, prot, MAP_SHARED | MAP_POPULATE, fd, aligned_offset));
            if (addr_ == static_cast<uint8_t *>(MAP_FAILED)) { 
                fprintf(stderr, "MMAP failed (errno %d): %s\n", errno, strerror(errno));
                throw std::runtime_error("mmap failed"); 
            }
        }

        ~FileMappingView() noexcept {
            if (addr_ != nullptr) {
                munmap(addr_, size_);
            }
        }

        FileMappingView(FileMappingView&& other) noexcept : FileView(other), addr_(other.addr_) {
            other.addr_ = nullptr;
            other.offset_ = 0;
            other.size_ = 0;
        }

#elif defined(_WIN32)
        FileMappingView() : FileView(0, 0), addr_(nullptr) {}

        FileMappingView(const HANDLE mapping_handle, size_t offset, size_t size, bool write = false): FileView(offset, size), addr_(nullptr) {
            // Do not call mmap if size == 0
            if (size == 0) {
                offset_ = 0;
                return;
            }

            const auto prot = write ? FILE_MAP_WRITE | FILE_MAP_READ : FILE_MAP_READ;

            const size_t aligned_offset = align_floor(offset, ALIGNED_GRANULARITY);
            const size_t aligned_size   = size + (offset - aligned_offset);

            offset_ = aligned_offset;
            size_   = aligned_size;


            {
                constexpr size_t offset_mask = 0xFFFFFFFF;
                const DWORD offset_high = (aligned_offset >> 32) & offset_mask;
                const DWORD offset_low = aligned_offset & offset_mask;
                addr_ = static_cast<uint8_t*>(MapViewOfFile(mapping_handle, prot, offset_high, offset_low, aligned_size));
                DWORD error = GetLastError();

                SPY_ASSERT(addr_ != nullptr, "Failed mmap file");
            }

            // PrefetchVirtualMemory is only present on Windows 8 and above, so we dynamically load it
            BOOL(WINAPI * pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            // may fail on pre-Windows 8 systems
            pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)> (GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

            if (pPrefetchVirtualMemory) {
                // advise the kernel to preload the mapped memory
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr_;
                range.NumberOfBytes = (SIZE_T)aligned_size;
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    SPY_ERROR_FMT("warning: PrefetchVirtualMemory failed: {}", system_error());
                }
            }
        }

        ~FileMappingView() noexcept {
            if (addr_ != nullptr) {
                if (!UnmapViewOfFile(addr_)) {
                    SPY_WARN("warning: UnmapViewOfFile failed");
                }
            }

        }


        FileMappingView(FileMappingView&& other) noexcept : FileView(other), addr_(other.addr_) {
            other.addr_ = nullptr;
            other.offset_ = 0;
            other.size_ = 0;
        }

        FileMappingView &operator = (FileMappingView &&other) noexcept {
            addr_ = other.addr_;
            offset_ = other.offset_;
            size_ = other.size_;

            other.addr_ = nullptr;
            other.offset_ = 0;
            other.size_ = 0;

            return *this;
        }
#endif

    public:
        void read(void *dst, size_t offset, size_t size) override {
            std::memcpy(dst, addr_ + offset - offset_, size);
        }

        void write(const void *src, size_t offset, size_t size) override {
            std::memcpy(addr_ + offset - offset_, src, size);
        }

        uint8_t *deref() const override { return addr_; }
        uint8_t *deref(size_t offset) const override { return addr_ + offset - offset_; }
    };


    struct FileStreamView : public FileView {
    protected:
        FILE* fp_;

        uint8_t* buffer_;

    public:
        FileStreamView() : fp_(nullptr), buffer_(nullptr) {}

        FileStreamView(FILE* fp, size_t offset, size_t size) : FileView(offset, size), fp_(fp) {
            buffer_ = new uint8_t[size];
        }

        FileStreamView(FileStreamView&& other) noexcept : FileView(other), fp_(other.fp_), buffer_(other.buffer_) {
            other.offset_ = 0;
            other.size_ = 0;
            other.fp_ = nullptr;
            other.buffer_ = nullptr;
        }

        ~FileStreamView() noexcept {
            delete[] buffer_; 
        }

    public:
        void read(void* dst, size_t offset, size_t size) override {
            std::fseek(fp_, offset, SEEK_SET);
            const size_t actual_size = fread(dst, size, 1, fp_);
            if (actual_size != size) {
                throw std::runtime_error("Fail freading");
            }
        }

        void write(const void* src, size_t offset, size_t size) override {
            std::fseek(fp_, offset, SEEK_SET);
            const size_t actual_size = fwrite(src, size, 1, fp_);
            if (actual_size != size) {
                throw std::runtime_error("Fail fwriting");
            }
        }

        uint8_t* deref() const override { return buffer_; }

        uint8_t* deref(size_t offset) const override { return buffer_ + offset - offset_; }
    };

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
					SPY_FATAL("Expect range pair not to be overlapped");
				}
				// Make fake entity
				auto iter_pair = view_map.insert({range_start, range_start});
				prev_iter = next_iter = iter_pair.first;
				++next_iter;
			}

			const size_t prev_end = prev_iter->second;

			// Check whether overlap
			if (prev_end > range_start) {
				SPY_FATAL("Expect range pair not to be overlapped");
			}
			if (next_iter != view_map.end()) {
				const size_t next_start = next_iter->first;
				if (range_end > next_start) {
					SPY_FATAL("Expect range pair not to be overlapped");
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
				SPY_FATAL("Cannot remove view from empty map");
			}

			auto prev_iter = view_map.lower_bound(range_start);
			if (prev_iter == view_map.begin()) {
				SPY_FATAL("Cannot remove view which hasn't been added");
			}

			--prev_iter;

			const size_t prev_start = prev_iter->first;
			const size_t prev_end   = prev_iter->second;

			if (prev_end < range_end) {
				SPY_FATAL("Cannot remove view spanning over multiple sub-view");
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

    class FileMappingViewFactory {
    private:
        /// Hold all views of file
        std::map<size_t, std::unique_ptr<FileView>> mapping_view_array_;

#ifdef __linux__
        /// The metadata of file
        FILE* fp_;
        int fd_;

    public:
        FileMappingViewFactory(): fp_(nullptr), fd_(-1) {}

        FileMappingViewFactory(FILE *fp, int fd): fp_(fp), fd_(fd) {}

        FileMappingViewFactory(FileMappingViewFactory &&other) noexcept: mapping_view_array_(std::move(other.mapping_view_array_)), fp_(other.fp_), fd_(other.fd_)  {
            other.fp_ = nullptr;
            other.fd_ = -1;
        }

        FileMappingViewFactory &operator =(FileMappingViewFactory &&other) noexcept {
            fp_ = other.fp_;
            fd_ = other.fd_;
            mapping_view_array_.clear();
            std::swap(mapping_view_array_, other.mapping_view_array_);
            return *this;
        }

    public:

        void set_fd(FILE *new_fp, int new_fd) { 
            // It is not need to release all views because the factory do not hold the ownership
            (void)new_fp;
            fd_ = new_fd; 
        }

#else
        /// The metadata of the file
        FILE* fp_;
        HANDLE mapping_handle_;

    public:
        FileMappingViewFactory() : fp_(nullptr), mapping_handle_(INVALID_HANDLE_VALUE) {}

        FileMappingViewFactory(FILE *fp, HANDLE fd) : fp_(fp), mapping_handle_(INVALID_HANDLE_VALUE) {
            set_fd(fp, fd);
        }

        FileMappingViewFactory(FileMappingViewFactory&& other) noexcept : mapping_view_array_(std::move(other.mapping_view_array_)), fp_(other.fp_), mapping_handle_(other.mapping_handle_) {
            other.fp_ = nullptr;
            other.mapping_handle_ = INVALID_HANDLE_VALUE;
        }

        FileMappingViewFactory& operator =(FileMappingViewFactory&& other) noexcept {
            fp_ = other.fp_;
            mapping_handle_ = other.mapping_handle_;

            other.fp_ = nullptr;
            other.mapping_handle_ = INVALID_HANDLE_VALUE;

            mapping_view_array_.clear();
            std::swap(mapping_view_array_, other.mapping_view_array_);
            return *this;
        }

        ~FileMappingViewFactory() noexcept {
            mapping_view_array_.clear();
            if (mapping_handle_ != INVALID_HANDLE_VALUE) {
                CloseHandle(mapping_handle_);
            }
        }

    public:

        void set_fd(FILE *new_fp, HANDLE new_fd) {
            fp_ = new_fp;

            /// It is necessary to release all views because the factory hold the ownership (mapping_handle) of them.
            if (!mapping_view_array_.empty()) {
                SPY_WARN("Trying to reassign file descriptor when mapping view is not empty");
                mapping_view_array_.clear();
            }
            if (mapping_handle_ != INVALID_HANDLE_VALUE) {
                CloseHandle(mapping_handle_);
            }
            mapping_handle_ = CreateFileMappingA(new_fd, NULL, PAGE_READONLY, 0, 0, NULL);
            DWORD error = GetLastError();
            SPY_ASSERT(mapping_handle_ != NULL, "Failed creating file mapping");
        }
#endif

    public:
        /*!
         * Creare a view of a portion of the file. It may use mapping or buffer for creating the view.
         * When the OS cannot allocate the address space of mapping, it turns to the buffer method.
         */
        void create_view(size_t offset, size_t size) {
            FileView* view_ptr = nullptr;
            try {
#ifdef _WIN32
                view_ptr = new FileMappingView{ mapping_handle_, offset, size };
#else
                view_ptr = new FileMappingView{ fd_, offset, size };
#endif
            }
            catch (const std::exception& err) {
                SPY_WARN("Failed creating mmap view, turn to stream view instead");
                view_ptr = new FileStreamView{ fp_, offset, size };
            }
            
            mapping_view_array_[offset] = std::unique_ptr<FileView>(view_ptr);
        }

    public:
        /*!
         * Get view by the offset of file. The position of view contains where the offset denotes.
         */
        std::unique_ptr<FileView> &get_view(size_t offset) { 
            auto iter = mapping_view_array_.lower_bound(offset);
            if (iter == mapping_view_array_.end() || iter->first != offset) { 
                if (iter == mapping_view_array_.begin()) {
                    SPY_FATAL("Cannot find view");
                }
                --iter; 
            }

            const auto &view = iter->second;
            if (view->offset() > offset || view->offset() + view->size() < offset) {
                SPY_FATAL("Cannot find view");
            }
            return iter->second;
        }
    };

} // namespace spy
