/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <iostream>
#include <string_view>
#include <argparse/argparse.hpp>

#include "util/shell/logger.h"

namespace spy {

	class Argument {

	private:
		argparse::ArgumentParser parser_;

	public:
		Argument(std::string_view name = "Spy") : parser_(name.data()) { 
			spy_info("Initialize command line arguments parser");
			define_augment();
		}

	public: /* Augment definition */
		void define_augment();

	public: /* Augment parsing */
		void parse_argv(int argc, char **argv);

		template<class T>
		T get_arg(std::string_view key) const { return parser_.get<T>(key); }

	};
	
}  // namespace spy