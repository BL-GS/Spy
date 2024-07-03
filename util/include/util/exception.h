/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstring>
#include <string>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
	#include <io.h>
#endif

#include <fmt/core.h>

namespace spy {

#ifdef _WIN32
	inline static auto system_error_code() {
		return GetLastError();
	}

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
	inline static auto system_error_code() {
		return errno;
	}

	inline static std::string system_error() {
		return fmt::format("[errno: {}]: {}", errno, strerror(errno));
	}
#endif

	class SpyException: public std::runtime_error {
	public:
		static constexpr std::string_view PREFIX			= "SpyException: ";
		static constexpr std::string_view UNKNOWN_EXCEPTION = "SpyException: Unknown exception";

	public:
		SpyException(): std::runtime_error(UNKNOWN_EXCEPTION.data()) {}

		SpyException(const std::string_view reason): std::runtime_error(PREFIX.data() + std::string(reason)) { }

		template<class ...Args>
		SpyException(const std::string_view &reason, Args ...args): 
			std::runtime_error(fmt::vformat(PREFIX.data() + std::string(reason), std::forward<Args>(args)...)) {}
	};

	class SpyAssertException: public SpyException {
	public:
		static constexpr std::string_view PREFIX			= "SpyAssertException: ";
		static constexpr std::string_view UNKNOWN_EXCEPTION = "SpyAssertException: Unknown exception";

	public:
		SpyAssertException(): SpyException(UNKNOWN_EXCEPTION.data()) {}

		SpyAssertException(const std::string_view &reason): SpyException(PREFIX.data() + std::string(reason)) {}

		template<class ...Args>
		SpyAssertException(const std::string_view &reason, Args ...args): 
			SpyException(PREFIX.data() + std::string(reason), std::forward<Args>(args)...) {}
	};

	class SpyUnimplementedException: public SpyException {
	public:
		static constexpr std::string_view PREFIX			= "SpyUnimplementedException: ";
		static constexpr std::string_view UNKNOWN_EXCEPTION = "SpyUnimplementedException: Unknown exception";

	public:
		SpyUnimplementedException(): SpyException(UNKNOWN_EXCEPTION.data()) {}

		SpyUnimplementedException(const std::string_view &reason): SpyException(PREFIX.data() + std::string(reason)) {}

		template<class ...Args>
		SpyUnimplementedException(const std::string_view &reason, Args ...args): 
			SpyException(PREFIX.data() + std::string(reason), std::forward<Args>(args)...) {}
	};

	class SpyOSException: public std::system_error {
	public:
		static constexpr std::string_view PREFIX			= "SpyOSException: ";
		static constexpr std::string_view UNKNOWN_EXCEPTION = "SpyOSException: Unknown exception";

	protected:
		std::string 	reason_;

	public:
		SpyOSException(): std::system_error(system_error_code(), std::system_category()), 
			reason_(std::string(UNKNOWN_EXCEPTION) + '\n' + std::system_error::what()) {}

		SpyOSException(int error_code): std::system_error(error_code, std::system_category()), 
			reason_(std::string(UNKNOWN_EXCEPTION) + '\n' + std::system_error::what()) {}

		SpyOSException(const std::string_view reason): 
			std::system_error(system_error_code(), std::system_category()), 
			reason_(PREFIX.data() + std::string(reason) + '\n' + std::system_error::what()) { }

		SpyOSException(int error_code, const std::string_view reason): 
			std::system_error(error_code, std::system_category()), 
			reason_(PREFIX.data() + std::string(reason) + '\n' + std::system_error::what()) { }

		template<class ...Args>
		SpyOSException(const std::string_view &reason, Args ...args): 
			std::system_error(system_error_code(), std::system_category()), 
			reason_(fmt::vformat(PREFIX.data() + std::string(reason), std::forward<Args>(args)...) + '\n' + std::system_error::what()) {}

		template<class ...Args>
		SpyOSException(int error_code, const std::string_view &reason, Args ...args): 
			std::system_error(error_code, std::system_category()), 
			reason_(fmt::vformat(PREFIX.data() + std::string(reason), std::forward<Args>(args)...) + '\n' + std::system_error::what()) {}

	public:	
		const char *what() const noexcept override { return reason_.c_str(); }
	};

}