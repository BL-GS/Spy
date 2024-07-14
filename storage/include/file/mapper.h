#pragma once

#include <cstddef>
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

#include "file/file_handler.h"
#include "file/file_view.h"

namespace spy {

    /*
     * To save memory of tensors which has been offloaded to GPU, and avoid dual memory occupation when loading model.
     */

#ifdef _WIN32

    class FileViewBuilder {
    protected:
        /// File handle for synchronous operations
        File    sync_file_;
        /// File handle for asynchronous operations
        File    async_file_;
        /// File handle for mmap
        HANDLE  mapping_handle_ = nullptr;

    public:
        FileViewBuilder() = default;

        ~FileViewBuilder() noexcept {
            if (mapping_handle_ != nullptr) { 
				const BOOL ret = CloseHandle(mapping_handle_); 
				spy_assert(ret == TRUE, "failed to close the mapping handle");
            }
        }

    public:
        void init_sync_handle(const std::string_view filename, bool write = false, bool existing = true, bool share = true) {
            const DWORD prot          = write     ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ;
            const DWORD share_flag    = share     ? (write ? FILE_SHARE_WRITE | FILE_SHARE_READ : FILE_SHARE_READ) : NULL;
            const DWORD disposition   = existing  ? OPEN_EXISTING : CREATE_ALWAYS;

            sync_file_ = File::make_file(filename, prot, share_flag, disposition, false);
        }

        void init_async_handle(const std::string_view filename, bool write = false, bool existing = true, bool share = true) {
            const DWORD prot          = write     ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ;
            const DWORD share_flag    = share     ? (write ? FILE_SHARE_WRITE | FILE_SHARE_READ : FILE_SHARE_READ) : NULL;
            const DWORD disposition   = existing  ? OPEN_EXISTING : CREATE_ALWAYS;

            async_file_ = File::make_file(filename, prot, share_flag, disposition, true);
        }

        void init_mapping(bool write = false) {
            const DWORD prot          = write     ? PAGE_READWRITE : PAGE_READONLY;

            if (!sync_file_.valid()) { throw SpyOSFileException("cannot create mapping on uninitialized file"); }

            mapping_handle_ = CreateFileMappingA(sync_file_.handle, nullptr, prot, 0, 0, nullptr);
            if (mapping_handle_ == nullptr) { throw SpyOSFileException("failed mapping file"); }
        }

    public:
        FileMappingView create_mapping_view(size_t size, int64_t offset, bool write = false) {
            if (mapping_handle_ == nullptr) { init_mapping(); }
            return { mapping_handle_, offset, size, write };
        }

        SyncFileView create_sync_file_view(size_t size, int64_t offset) const {
            if (!sync_file_.valid()) { throw SpyOSFileException("cannot create view on uninitialized file"); }
            return { sync_file_.handle, offset, size };
        }

        ASyncFileView create_async_file_view(size_t size, int64_t offset) const {
            if (!async_file_.valid()) { throw SpyOSFileException("cannot create view on uninitialized file"); }
            return { async_file_.handle, offset, size };
        }
    };

#else

    class FileViewBuilder {
    protected:
        /// File handle
        File file_;

    public:
        FileViewBuilder() = default;

        ~FileViewBuilder() noexcept = default;

    public:
        void init_sync_handle(const std::string_view filename, bool write = false, bool existing = true, bool share = true) {
            if (!file_.valid()) { init_handle(filename, write, existing, share); }
        }

        void init_async_handle(const std::string_view filename, bool write = false, bool existing = true, bool share = true) {
            if (!file_.valid()) { init_handle(filename, write, existing, share); }
        }

        void init_mapping(bool write = false) {
            if (!file_.valid()) { throw SpyOSFileException("cannot create mapping on uninitialized file"); }
        }

    public:
        FileMappingView create_mapping_view(size_t size, int64_t offset, bool write = false) {
            if (!file_.valid()) { throw SpyOSFileException("cannot create view on uninitialized file"); }
            return { file_.descriptor, offset, size, write };
        }

        SyncFileView create_sync_file_view(size_t size, int64_t offset) const {
            if (!file_.valid()) { throw SpyOSFileException("cannot create view on uninitialized file"); }
            return { file_.descriptor, offset, size };
        }

        ASyncFileView create_async_file_view(size_t size, int64_t offset) const {
            if (!file_.valid()) { throw SpyOSFileException("cannot create view on uninitialized file"); }
            return { file_.descriptor, offset, size };
        }

    private:
        void init_handle(const std::string_view filename, bool write, bool existing, [[maybe_unused]] bool share) {
            const int prot = write      ? O_RDWR : O_RDONLY;
            const int flag = existing   ? 0      : O_CREAT;

            file_ = File::make_file(filename, prot | flag, 0);
        }
    };

#endif // _WIN32

}  // namespace spy
