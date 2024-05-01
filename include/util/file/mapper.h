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

#include "util/logger.h"
#include "util/file/file_handler.h"
#include "util/file/file_view.h"

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
				SPY_ASSERT_NOEXCEPTION(ret == TRUE, "failed to close the mapping handle");
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


    class FileViewFactory final: FileViewBuilder {
    protected:
        std::map<size_t, std::unique_ptr<FileView>> view_map_;

    public:
        /*!
         * @details Creare a view of a portion of the file. It may use mapping or buffer for creating the view.
         * When the OS cannot allocate the address space of mapping, it turns to the buffer method.
         */
        template<class T_View>
        void create_view(size_t offset, size_t size) {
            view_map_[offset] = std::unique_ptr<T_View>(offset, size);
        }

        void create_view(size_t offset, size_t size, bool write = false) {
            view_map_[offset] = 
                std::make_unique<FileMappingView>(sync_file_.handle, offset, size, write);
        }

        /*!
         * @brief view by the offset of file. The position of view contains where the offset denotes.
         */
        std::unique_ptr<FileView> &get_view(size_t offset) {
            auto iter = view_map_.lower_bound(offset);
            if (iter == view_map_.end() || iter->first != offset) { 
                if (iter == view_map_.begin()) {
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
    
#else

    

#endif // _WIN32

}  // namespace spy
