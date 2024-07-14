#pragma once

#ifdef __has_include
	#if __has_include(<unistd.h>)
		#include <unistd.h>
		#if defined(_POSIX_MAPPED_FILES)
			#include <fcntl.h>
			#include <sys/mman.h>
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
	#include <io.h>
	#include <windows.h>
#endif

#include "util/shell/logger.h"
#include "util/exception.h"
#include "file/exception.h"

namespace spy {

#ifdef _WIN32
	struct File {
	public:
		HANDLE handle;

	public:
		File(HANDLE handle = INVALID_HANDLE_VALUE): handle(handle) {}

		File(File &&other) noexcept {
			if (valid()) {
				const BOOL ret = CloseHandle(handle); 
				spy_assert(ret == TRUE, "failed to close the file handle");
			}
			handle = other.handle;
			other.handle = INVALID_HANDLE_VALUE;
		}

		~File() noexcept { 
			if (valid()) { 
				const BOOL ret = CloseHandle(handle); 
				spy_assert(ret == TRUE, "failed to close the file handle");
			}
		}

		File &operator=(File &&other) noexcept { 
			if (valid()) {
				const BOOL ret = CloseHandle(handle); 
				spy_assert(ret == TRUE, "failed to close the file handle");
			}
			handle = other.handle;
			other.handle = INVALID_HANDLE_VALUE;
			return *this;
		}

	public:
		/*!
		 * @brief Whether the handle of file is valid
		 */
		bool valid() const { return handle != INVALID_HANDLE_VALUE;  }

		/*!
		 * @brief Reset the handle to a empty value
		 * @return false if the handle if invalid, otherwise true.
		 * @throw SpyOSException if failed to close a valid handle
		 */
		bool reset(HANDLE new_handle = INVALID_HANDLE_VALUE) {
			if (valid()) { // close the handle if valid
				const BOOL ret = CloseHandle(handle); 
				if (ret == FALSE) {
					throw SpyOSException("failed to close the file handle");
				}
				handle = new_handle;
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
		int64_t seek(const int64_t offset, const DWORD method = FILE_CURRENT) const {
			LARGE_INTEGER offset_integer { .QuadPart = offset };
			offset_integer.LowPart = SetFilePointer(handle, offset_integer.LowPart, &offset_integer.HighPart, method);
			if (offset_integer.LowPart == INVALID_SET_FILE_POINTER) {
				throw SpyOSFileException("failed to seek file");
			}
			return offset_integer.QuadPart;
		}

		/*!
		 * @brief Truncate or expand the file to the current file pointer.
		 * @note The data pointer after truncate is undefined, use seek if needed.
		 */
		void truncate() const {
			const auto ret = SetEndOfFile(handle);
			if (ret == 0) {
				throw SpyOSFileException("failed to truncate file");
			}
		}

		/*!
		 * @brief Truncate or expand the file to the specific size.
		 * @note The data pointer after truncate is undefined, use seek if needed.
		 */
		void truncate(const int64_t size) const {
			seek(size, FILE_BEGIN);
			truncate();
		}

	public:
		/*!
		 * @brief Get the size of the file
		 */
		int64_t size() const {
			LARGE_INTEGER size;
			WINBOOL ret = GetFileSizeEx(handle, &size);
			if (ret == 0) {
				throw SpyOSFileException("failed to get size of file");
			}
			return size.QuadPart;
		}

		std::string name() const {
			TCHAR buffer[MAX_PATH];
			DWORD ret = GetFinalPathNameByHandle(handle, buffer, MAX_PATH, VOLUME_NAME_NT);
			if (ret == 0) {
				throw SpyOSFileException("failed to get the name of file");
			}
			if (ret >= MAX_PATH) {
				std::string res(ret, '\0');
				GetFinalPathNameByHandle(handle, res.data(), MAX_PATH, VOLUME_NAME_NT);
			}
			return { buffer };
		}

	public:
		static File make_file(std::string_view filename, DWORD access, DWORD share_flag,
			DWORD disposition, const bool overlapped) {
			HANDLE new_handle = CreateFile(filename.data(), 
				access, 
				share_flag, 
				nullptr, 
				disposition,
				overlapped ? NULL : FILE_FLAG_OVERLAPPED,
				nullptr
			);
			if (new_handle == INVALID_HANDLE_VALUE) {
				throw SpyOSFileException("failed creating file");
			}
			return { new_handle };
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
				const int ret = close(descriptor);
				spy_assert(ret == 0, "failed to close the file handle");
			}
			descriptor 			= other.descriptor;
			other.descriptor 	= INVALID_FILE_DESCRIPTOR;
		}

		~File() noexcept { 
			if (valid()) { 
				const int ret = close(descriptor);
				spy_assert(ret == 0, "failed to close the file handle");
			}
		}

		File &operator=(File &&other) noexcept { 
			if (valid()) {
				const int ret = close(descriptor);
				spy_assert(ret == 0, "failed to close the file handle");
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
				if (ret != 0) { throw SpyOSException("failed to close the file handle"); }
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
				throw SpyOSFileException("failed to seek file");
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
				throw SpyOSFileException("failed to truncate file");
			}
		}

		/*!
		 * @brief Truncate or expand the file to the specific size.
		 * @note The data pointer after truncate is undefined, use seek if needed.
		 */
		void truncate(const int64_t size) const {
			const int ret = ftruncate(descriptor, size);
			if (ret != 0) {
				throw SpyOSFileException("failed to truncate file");
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
			const std::string info_file = fmt::format("/proc/self/fd/{}", descriptor);
			char file_path[256] = {'\0'};
			int ret = readlink(info_file.c_str(),file_path,sizeof(file_path) - 1);

			if (ret == -1) {
				throw SpyOSFileException("failed to get the name of file");
			}
			return { file_path, file_path + ret };
		}

	public:
		static File make_file(std::string_view filename, int flag, int mode) {
			int new_descriptor = open(filename.data(), flag, mode);
			if (new_descriptor == -1) {
				throw SpyOSFileException("failed creating file");
			}
			return { new_descriptor };
		}
	};

#endif

} // namespace spy