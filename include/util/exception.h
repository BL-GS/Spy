/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <stdexcept>
#include <string>

namespace spy {

	class SpyException: std::exception {
	protected:
		std::string reason_;

	public:
		SpyException(): reason_("SpyException: Unknown exception") {}
		SpyException(const std::string_view reason): reason_("SpyException: ") { reason_ += reason; }

	public:
		const char * what() const noexcept override { return reason_.c_str(); }
	};

	class SpyNumericException: SpyException {
	public:
		SpyNumericException(): SpyException("SpyNumericException: Unknown exception") {}
		SpyNumericException(const std::string_view reason): SpyException(std::string("SpyNumericException") + reason.data()) {}
	};

	class SpyAssertException: SpyException {
	public:
		SpyAssertException(): SpyException("SpyAssertException: Unknown exception") {}
		SpyAssertException(const std::string_view reason): SpyException(std::string("SpyAssertException") + reason.data()) {}
	};

	class SpyOSException: SpyException {
	public:
		SpyOSException(): SpyException("SpyOSException: Unknown exception") {}
		SpyOSException(const std::string_view reason): SpyException(std::string("SpyOSException") + reason.data()) {}
	};

	class SpyNoneException: SpyException {
	public:
		SpyNoneException(): SpyException("SpyNoneException: Unknown exception") {}
		SpyNoneException(const std::string_view reason): SpyException(std::string("SpyNoneException") + reason.data()) {}
	};

	class SpyUnimplementedException: SpyException {
	public:
		SpyUnimplementedException(): SpyException("SpyUnimplementedException: Unknown exception") {}
		SpyUnimplementedException(const std::string_view reason): SpyException(std::string("SpyUnimplementedException") + reason.data()) {}
	};

}