/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <exception>
#include <string>

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
	#include <io.h>
#endif

namespace spy {

#ifdef _WIN32
	inline static std::string system_error() {
		DWORD err = GetLastError();
		LPSTR buf;
		size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&buf), 0, nullptr);
		if (size == 0) {
			return "FormatMessageA failed";
		}
		std::string ret(buf, size);
		LocalFree(buf);

		return ret;
	}
#else
	inline static std::string system_error() {
		return fmt::format("[errno: {}]: {}", errno, strerror(errno));
	}
#endif

	class SpyException: public std::exception {
	protected:
		std::string reason_;

	public:
		SpyException(): reason_("SpyException: Unknown exception") {}
		SpyException(const std::string &reason): reason_("SpyException: " + reason) {}

	public:
		const char * what() const noexcept override { return reason_.c_str(); }
	};

	class SpyNumericException: public SpyException {
	public:
		SpyNumericException(): SpyException("SpyNumericException: Unknown exception") {}
		SpyNumericException(const std::string &reason): SpyException("SpyNumericException" + reason) {}
	};

	class SpyAssertException: public SpyException {
	public:
		SpyAssertException(): SpyException("SpyAssertException: Unknown exception") {}
		SpyAssertException(const std::string &reason): SpyException("SpyAssertException" + reason) {}
	};

	class SpyNoneException: public SpyException {
	public:
		SpyNoneException(): SpyException("SpyNoneException: Unknown exception") {}
		SpyNoneException(const std::string &reason): SpyException("SpyNoneException" + reason) {}
	};

	class SpyUnimplementedException: public SpyException {
	public:
		SpyUnimplementedException(): SpyException("SpyUnimplementedException: Unknown exception") {}
		SpyUnimplementedException(const std::string &reason): SpyException("SpyUnimplementedException" + reason) {}
	};

	class SpyOSException: public std::system_error {
	public:
		SpyOSException(): std::system_error(errno, std::system_category()) {}
		SpyOSException(const std::string &reason): std::system_error(errno, std::system_category(), reason) {}
	};

}