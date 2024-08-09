#pragma once

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <stdexcept>
#include <system_error>
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

#include "file_view.h"

namespace spy {

    /*
     * To save memory of tensors which has been offloaded to GPU, and avoid dual memory occupation when loading model.
     */

#ifdef _WIN32

    class FileViewBuilder {
    protected:
        HANDLE sync_file_handle_;
        HANDLE async_file_handle_;

        HANDLE mapping_handle_;

    public:
        FileViewBuilder(): sync_file_handle_(INVALID_HANDLE_VALUE), async_file_handle_(INVALID_HANDLE_VALUE), mapping_handle_(INVALID_HANDLE_VALUE) { }

        ~FileViewBuilder() {
            if (mapping_handle_ != INVALID_HANDLE_VALUE)            { CloseHandle(mapping_handle_);     }
            if (async_file_handle_    != INVALID_HANDLE_VALUE)      { CloseHandle(async_file_handle_);  }
            if (sync_file_handle_ != INVALID_HANDLE_VALUE)          { CloseHandle(sync_file_handle_);   }
        }

        FileViewBuilder(FileViewBuilder &&other) noexcept: sync_file_handle_(other.sync_file_handle_), async_file_handle_(other.async_file_handle_), mapping_handle_(other.mapping_handle_) {
            other.async_file_handle_ = INVALID_HANDLE_VALUE;
            other.sync_file_handle_ = INVALID_HANDLE_VALUE;
            other.mapping_handle_   = INVALID_HANDLE_VALUE;
        } 

    public:
        bool open_if_exist(const std::string &filename, bool write = false, bool use_overlapped = false) {
            const DWORD prot          = write ?           GENERIC_READ | GENERIC_WRITE        : GENERIC_READ;
            const DWORD share_flag    = write ?           FILE_SHARE_READ | FILE_SHARE_WRITE  : FILE_SHARE_READ;
            const DWORD flag          = use_overlapped ?  FILE_FLAG_OVERLAPPED                : NULL;
            const DWORD open_flag     = OPEN_EXISTING;

            HANDLE file_handle = CreateFile(filename.c_str(), prot, share_flag, nullptr, open_flag, flag, nullptr);
            if (file_handle_ == INVALID_HANDLE_VALUE) {
                fprintf(stderr, "Failed reopening file: %s\n", llama_format_win_err().c_str());
                return false;
            }

            if (flag != NULL) {
                async_file_handle_ = file_handle;
            } else {
                sync_file_handle_ = file_handle;
            }

            return true;
        }

        void init_mapping() {
            mapping_handle_ = CreateFileMappingA(sync_file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);
            if (mapping_handle_ == nullptr) { 
                fprintf(stderr, "Failed mapping file: %s\n", llama_format_win_err().c_str());
            }
        }

    public:
        FileMappingView create_mapping_view(size_t size, int64_t offset, bool write = false) {
            if (mapping_handle_ == INVALID_HANDLE_VALUE) { init_mapping(); }
            return { mapping_handle_, offset, size, write };
        }

        SyncFileView create_sync_file_view(size_t size, int64_t offset) const {
            return { sync_file_handle_, offset, size };
        }

        ASyncFileView create_async_file_view(size_t size, int64_t offset) const {
            return { async_file_handle_, offset, size };
        }
    };


    class FileViewFactory final: FileViewBuilder {
    protected:
        std::map<size_t, std::unique_ptr<FileView>> view_map_;

    public:
        /*!
         * Creare a view of a portion of the file. It may use mapping or buffer for creating the view.
         * When the OS cannot allocate the address space of mapping, it turns to the buffer method.
         */
        template<class T_View>
        void create_view(int64_t offset, size_t size) {
            view_map_[offset] = std::unique_ptr<T_View>(offset, size);
        }

        void create_view(int64_t offset, size_t size, bool write = false) {
            view_map_[offset] = 
                std::make_unique<FileMappingView>(sync_file_handle_, offset, size, write);
        }

        /*!
         * Get view by the offset of file. The position of view contains where the offset denotes.
         */
        std::unique_ptr<FileView> &get_view(size_t offset) {
            auto iter = view_map_.lower_bound(offset);
            if (iter == view_map_.end() || iter->first != offset) { 
                if (iter == view_map_.begin()) {
                    GGML_ASSERT(false, "Cannot find view");
                }
                --iter; 
            }

            const auto &view = iter->second;
            if (view->offset() > offset || view->offset() + view->size() < offset) {
                GGML_ASSERT(false, "Cannot find view");
            }
            return iter->second;
        }
    };
    
#else

    struct File {
	public:
		static constexpr int INVALID_FILE_DESCRIPTOR = -1;
	public:
		int descriptor;

	public:
		File(int descriptor = INVALID_FILE_DESCRIPTOR): descriptor(descriptor) {}

		File(File &&other) noexcept {
			if (valid()) {
				close(descriptor);
			}
			descriptor 			= other.descriptor;
			other.descriptor 	= INVALID_FILE_DESCRIPTOR;
		}

		~File() noexcept { 
			if (valid()) { 
				close(descriptor);
			}
		}

		File &operator=(File &&other) noexcept { 
			if (valid()) {
				close(descriptor);
			}
			descriptor 			= other.descriptor;
			other.descriptor 	= INVALID_FILE_DESCRIPTOR;
			return *this;
		}

	public:
		/*!
		 * @brief Whether the handle of file is valid
		 */
		bool valid() const { return descriptor != INVALID_FILE_DESCRIPTOR;  }

		/*!
		 * @brief Reset the handle to a empty value
		 * @return false if the handle if invalid, otherwise true.
		 * @throw SpyOSException if failed to close a valid handle
		 */
		bool reset(int new_descriptor = INVALID_FILE_DESCRIPTOR) {
			if (valid()) { // close the handle if valid
				const int ret = close(descriptor);
				if (ret != 0) { 
                    fprintf(stderr, "failed to close the file handle");
                    throw std::system_error(errno, std::system_category()); 
                }
				descriptor = new_descriptor;
				return true;
			}
			return false;
		}

	public:
		/*!
		 * @brief Move the pointer of file
		 * @param offset the offset of pointer
		 * @param method the method for setting pointer, default as FILE_CURRENT
		 * @return The absolute offset of the current file pointer
		 */
		int64_t seek(const int64_t offset, const int whence = SEEK_CUR) const {
			const int64_t cur_offset = lseek(descriptor, offset, whence);
			if (cur_offset == -1) {
                fprintf(stderr, "failed to seek file");
				throw std::system_error(errno, std::system_category());
			}
			return cur_offset;
		}

		/*!
		 * @brief Truncate or expand the file to the current file pointer.
		 * @note The data pointer after truncate is undefined, use seek if needed.
		 */
		void truncate() const {
			const int64_t cur_offset = lseek(descriptor, 0, SEEK_CUR);
			const int ret = ftruncate(descriptor, cur_offset);
			if (ret != 0) {
                fprintf(stderr, "failed to truncate file");
				throw std::system_error(errno, std::system_category());
			}
		}

		/*!
		 * @brief Truncate or expand the file to the specific size.
		 * @note The data pointer after truncate is undefined, use seek if needed.
		 */
		void truncate(const int64_t size) const {
			const int ret = ftruncate(descriptor, size);
			if (ret != 0) {
                fprintf(stderr, "failed to truncate file");
				throw std::system_error(errno, std::system_category());
			}
		}

	public:
		/*!
		 * @brief Get the size of the file
		 */
		int64_t size() const {
			const int64_t origin_offset = seek(0, SEEK_CUR);
			const int64_t res = seek(0, SEEK_END);
			seek(origin_offset, SEEK_SET);
			return res;
		}

		std::string name() const {
			const std::string info_file("/proc/self/fd/" + std::to_string(descriptor));
			char file_path[256] = {'\0'};
			int ret = readlink(info_file.c_str(),file_path,sizeof(file_path) - 1);

			if (ret == -1) {
                fprintf(stderr, "failed to get the name of file");
				throw std::system_error(errno, std::system_category());
			}
			return { file_path, file_path + ret };
		}

	public:
		static File make_file(std::string_view filename, int flag, int mode) {
			int new_descriptor = open(filename.data(), flag, mode);
			if (new_descriptor == -1) {
                fprintf(stderr, "failed creating file");
                return { INVALID_FILE_DESCRIPTOR };
			}
			return { new_descriptor };
		}
	};


    class FileViewBuilder {
    protected:
        /// File handle
        File file_;

    public:
        FileViewBuilder() = default;

        FileViewBuilder(FileViewBuilder &&other) noexcept = default;

        ~FileViewBuilder() noexcept = default;

    public:
        bool open_if_exist(const std::string &filename, bool write = false, [[maybe_unused]] bool use_overlapped = false) {
            init_handle(filename, write, true, false);
            return file_.valid();
        }

        void init_sync_handle(const std::string_view filename, bool write = false, bool existing = true, bool share = true) {
            if (!file_.valid()) { init_handle(filename, write, existing, share); }
        }

        void init_async_handle(const std::string_view filename, bool write = false, bool existing = true, bool share = true) {
            if (!file_.valid()) { init_handle(filename, write, existing, share); }
        }

        void init_mapping([[maybe_unused]] bool write = false) {
            if (!file_.valid()) { 
                fprintf(stderr, "cannot create mapping on uninitialized file");
                throw std::system_error(errno, std::system_category()); 
            }
        }

    public:
        FileMappingView create_mapping_view(size_t size, int64_t offset, bool write = false) {
            if (!file_.valid()) { 
                fprintf(stderr, "cannot create view on uninitialized file");
                throw std::system_error(errno, std::system_category()); 
            }
            return { file_.descriptor, offset, size, write };
        }

        SyncFileView create_sync_file_view(size_t size, int64_t offset) const {
            if (!file_.valid()) { 
                fprintf(stderr, "cannot create view on uninitialized file");
                throw std::system_error(errno, std::system_category()); 
            }
            return { file_.descriptor, offset, size };
        }

        ASyncFileView create_async_file_view(size_t size, int64_t offset) const {
            if (!file_.valid()) { 
                fprintf(stderr, "cannot create view on uninitialized file");
                throw std::system_error(errno, std::system_category()); 
            }
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
