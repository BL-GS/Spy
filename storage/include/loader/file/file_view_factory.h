#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <memory>
#include <map>
#include <fstream>

#include "file_view.h"

namespace spy {

    class FileMappingViewFactory {
    private:
        /// Hold all views of file
        std::map<size_t, std::unique_ptr<FileMappingView>> mapping_view_array_;

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
                printf("Warning: Trying to reassign file descriptor when mapping view is not empty\n");
                mapping_view_array_.clear();
            }
            if (mapping_handle_ != INVALID_HANDLE_VALUE) {
                CloseHandle(mapping_handle_);
            }
            mapping_handle_ = CreateFileMappingA(new_fd, NULL, PAGE_READONLY, 0, 0, NULL);
            DWORD error = GetLastError();
            if (mapping_handle_ == NULL) { throw std::runtime_error(std::string("CreateFileMappingA failed: ") + llama_format_win_err(error)); }
        }
#endif

    public:
        /*!
         * Creare a view of a portion of the file. It may use mapping or buffer for creating the view.
         * When the OS cannot allocate the address space of mapping, it turns to the buffer method.
         */
        void create_view(size_t offset, size_t size) {
            std::unique_ptr<FileMappingView> view_ptr;
#ifdef _WIN32
            view_ptr = std::make_unique<FileMappingView>(mapping_handle_, offset, size);
#else
            view_ptr = std::make_unique<FileMappingView>(fd_, offset, size);
#endif
            mapping_view_array_[offset] = std::move(view_ptr);
        }

    public:
        /*!
         * Get view by the offset of file. The position of view contains where the offset denotes.
         */
        std::unique_ptr<FileMappingView> &get_view(size_t offset) { 
            auto iter = mapping_view_array_.lower_bound(offset);
            if (iter == mapping_view_array_.end() || iter->first != offset) { 
                if (iter == mapping_view_array_.begin()) {
                    throw std::runtime_error("Cannot find view");
                }
                --iter; 
            }

            return iter->second;
        }
    };

} // namespace spy
