#pragma once

#include <fstream>
#include <cstring>
#include <memory>
#include <string_view>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "util/align.h"
#include "model/type.h"
#include "model/file/type.h"
#include "model/file/config.h"

/* Class Definition */
namespace spy {

	class GGUFLoader {
	public:
		static constexpr auto   GGUF_MAGIC 		= GGUFContext::GGUF_MAGIC;
		static constexpr size_t GGUF_MAGIC_SIZE = GGUF_MAGIC.size();

	public:
		static std::pair<std::unique_ptr<GGUFContext>, ModelType> init_from_file(const std::string_view filename) {
			auto      context_ptr = std::make_unique<GGUFContext>();
			auto      &context  = *context_ptr;
			ModelType model_type  = ModelType::ModelTypeEnd;

			std::fstream file(filename.data(), std::ios::binary | std::ios::in);
			spy_assert(file.is_open(), "Cannot open file: {}", filename);

			/* Read magic */
			try {
				read_gguf_array(file, context.header.magic.data(), GGUF_MAGIC_SIZE);
				spy_assert(context.header.magic == GGUF_MAGIC,
				           "Invalid magic characters (given: {}, expect: {})",
				           std::string_view(context.header.magic.data(), GGUF_MAGIC_SIZE), 
						   std::string_view(GGUF_MAGIC.data(), GGUF_MAGIC_SIZE));
			} catch (const std::exception &err) {
				spy_fatal("Failed reading gguf magic from file {:32}: {}", filename, err.what());
			}

			/* Read header */
			try {
				read_gguf_value(file, context.header.version);
				read_gguf_value(file, context.header.num_tensor);
				read_gguf_value(file, context.header.num_kv);
				spy_assert(context.header.num_tensor < std::numeric_limits<uint64_t>::max() / 2 / sizeof(GGUFTensorInfo));
				spy_assert(context.header.num_kv     < std::numeric_limits<uint64_t>::max() / 2 / sizeof(GGUFValue));
			} catch (const std::exception &err) {
				spy_fatal("Failed reading gguf header from file {}: {}", filename, err.what());
			}

			/* Read kv pairs */
			try {
				const size_t num_kv = context.header.num_kv;

				for (size_t i = 0; i < num_kv; ++i) {
					const std::string  cur_key  = read_gguf_string(file);
					const GGUFDataType cur_type = read_gguf_type(file);

					spy_info("Load kv pair: {:<48} - {:8}", cur_key, cur_type);

					context.add_gguf_value(cur_key, read_gguf_kv(file, cur_type));
				}
			} catch (const std::exception &err) {
				spy_fatal("Failed reading gguf kv pairs from file {}: {}", filename, err.what());
			}

			try {
				// Cannot fetch architecture name by LLMKey
				const GGUFValue &arch_kv  = context.find_gguf_value(LLMKey::GENERAL_ARCHITECTURE);
				spy_assert(arch_kv.get_type() == GGUFDataType::String, "Expect the architecture value to be string");
				context.arch_name = arch_kv.get_value<std::string>();
			} catch (const std::exception &err) {
				spy_fatal("Failed reading gguf arch name from file {}: {}", filename, err.what());
			}

			/* Read tensor info */
			try {
				const size_t num_tensor = context.header.num_tensor;

				for (size_t tensor_idx = 0; tensor_idx < num_tensor; ++tensor_idx) {
					GGUFTensorInfo cur_info;
					cur_info.num_element.fill(1UL);
					const std::string tensor_name = read_gguf_string(file);
					read_gguf_value(file, cur_info.num_dim);
					read_gguf_array(file, cur_info.num_element.data(), cur_info.num_dim);
					read_gguf_value(file, cur_info.type);
					read_gguf_value(file, cur_info.offset);

					context.infos[tensor_name] = cur_info;
				}
			} catch (const std::exception &err) {
				spy_fatal("Failed reading gguf tensor infos from file {}: {}", filename, err.what());
			}

			/* Read offset information */
			try {
				const auto alignment_option = context.find_gguf_value_option(LLMKey::GENERAL_ALIGNMENT);
				if (alignment_option.has_value()) { context.alignment = alignment_option.value().get_value<uint32_t>(); }

				const size_t offset = align_ceil<size_t>(file.tellg(), context.alignment);
				file.seekg(offset);
				context.offset = offset;
			} catch (const std::exception &err) {
				spy_fatal("Failed reading gguf offset information from file {}: {}", filename, err.what());
			}

			/* Calculate the total size of the data section */
			for (const auto &cur_info_pair: context.infos) {
				const GGUFTensorInfo &cur_info = cur_info_pair.second;
				
				size_t num_element = 1;
				for (auto dim : cur_info.num_element) { num_element *= dim; }

				const size_t block_size = get_block_size(cur_info.type);
				spy_assert(num_element % block_size == 0, 
					"tensor: number of {} elements({}) is not a multiple of block size()", 
					cur_info.type, num_element, block_size);

				const size_t cur_size = get_row_size(cur_info.type, num_element);
				context.size 		 += align_ceil(cur_size, context.alignment);
			}

			model_type         	       = get_arch_type_from_name(context.arch_name);
			context.model_tensor_table = TensorNameTable(model_type);

			return { std::move(context_ptr), model_type };
		}

	public: /* Utilities */
		template<class T>
		static void read_gguf_array(std::fstream &file, T *addr, size_t num, size_t size = sizeof(T)) {
			file.read(reinterpret_cast<char *>(addr), num * size);
		}

		template<class T>
		static T read_gguf_value(std::fstream &file) {
			T res;
			file.read(reinterpret_cast<char *>(&res), sizeof(T));
			return res;
		}

		template<class T>
		static void read_gguf_value(std::fstream &file, T &value) {
			file.read(reinterpret_cast<char *>(&value), sizeof(T));
		}

		static std::string read_gguf_string(std::fstream &file) {
			const auto len = read_gguf_value<uint64_t>(file);
			std::string res(len, '\0');
			file.read(res.data(), len);
			return res;
		}

		static GGUFDataType read_gguf_type(std::fstream &file) {
			GGUFDataType res = GGUFDataType::GGUFDataTypeEnd;
			file.read(reinterpret_cast<char *>(&res), sizeof(GGUFDataType));
			return res;
		}

		static GGUFValue read_gguf_kv(std::fstream &file, GGUFDataType type) {
			switch(type) {
				case GGUFDataType::UInt8:   return { read_gguf_value<uint8_t>(file)     };
				case GGUFDataType::Int8:    return { read_gguf_value<int8_t>(file)      };
				case GGUFDataType::UInt16:  return { read_gguf_value<uint16_t>(file)    };
				case GGUFDataType::Int16:   return { read_gguf_value<int16_t>(file)     };
				case GGUFDataType::UInt32:  return { read_gguf_value<uint32_t>(file)    };
				case GGUFDataType::Int32:   return { read_gguf_value<int32_t>(file)     };
				case GGUFDataType::Float32: return { read_gguf_value<float>(file)       };
				case GGUFDataType::Bool:    return { read_gguf_value<bool>(file)     	};
				case GGUFDataType::UInt64:  return { read_gguf_value<uint64_t>(file)    };
				case GGUFDataType::Int64:   return { read_gguf_value<int64_t>(file)     };
				case GGUFDataType::Float64: return { read_gguf_value<double>(file)      };
				case GGUFDataType::String:  return { read_gguf_string(file)     		};
				case GGUFDataType::Array:   {
					const auto arr_type = read_gguf_type(file);
					const auto arr_len  = read_gguf_value<uint64_t>(file);
					GGUFArray array;
					array.reserve(arr_len);
					for (uint64_t i = 0; i < arr_len; ++i) { 
						switch(arr_type) {
							case GGUFDataType::UInt8:   array.emplace_back(read_gguf_value<uint8_t>(file));  break;
							case GGUFDataType::Int8:    array.emplace_back(read_gguf_value<int8_t>(file));   break;
							case GGUFDataType::UInt16:  array.emplace_back(read_gguf_value<uint16_t>(file)); break;
							case GGUFDataType::Int16:   array.emplace_back(read_gguf_value<int16_t>(file));  break;
							case GGUFDataType::UInt32:  array.emplace_back(read_gguf_value<uint32_t>(file)); break;
							case GGUFDataType::Int32:   array.emplace_back(read_gguf_value<int32_t>(file));	 break;
							case GGUFDataType::Float32: array.emplace_back(read_gguf_value<float>(file));	 break;
							case GGUFDataType::Bool:    array.emplace_back(read_gguf_value<bool>(file));	 break;
							case GGUFDataType::UInt64:  array.emplace_back(read_gguf_value<uint64_t>(file)); break;
							case GGUFDataType::Int64:   array.emplace_back(read_gguf_value<int64_t>(file));  break;
							case GGUFDataType::Float64: array.emplace_back(read_gguf_value<double>(file));	 break;
							case GGUFDataType::String:  array.emplace_back(read_gguf_string(file));			 break;
							default:
								spy_assert(false, "Unknown gguf data type in array");
						}
					}
					return array;
				}
				default:
					spy_assert(false, "Unknown gguf data type");
			}
			return {};
		}

	};

}  // namespace spy