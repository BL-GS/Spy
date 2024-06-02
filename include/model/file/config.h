#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <variant>
#include <map>
#include <optional>

#include "number/tensor.h"
#include "model/config.h"
#include "model/file/type.h"

namespace spy {

	using GGUFArray   = std::vector<std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
								float, bool, std::string, uint64_t, int64_t, double>>;
	using GGUFElement = std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float, bool,
								std::string, GGUFArray, uint64_t, int64_t, double>;

	template<GGUFDataType T_type>
	using GGUFTypeMap = std::variant_alternative_t<static_cast<size_t>(T_type), GGUFElement>;

	struct GGUFValue {
	public:
		GGUFElement value;

	public:
		GGUFValue() = default;

		template<class T>
		GGUFValue(T &&new_val): value(std::forward<T>(new_val)) {}

	public:
		GGUFDataType get_type() const { 
			return static_cast<GGUFDataType>(value.index()); }

		template<class T>
		T get_value() const { return std::get<T>(value); }
	};

	struct GGUFHeader {
		static constexpr size_t GGUF_MAGIC_SIZE = 4;
		using MagicArray = std::array<char, GGUF_MAGIC_SIZE>;

		static constexpr MagicArray GGUF_MAGIC{ 'G', 'G', 'U', 'F'};

		MagicArray magic;

		uint32_t version;
		uint64_t num_tensor;
		uint64_t num_kv;

		constexpr GGUFHeader() : magic(GGUF_MAGIC), version(3), num_tensor(0), num_kv(0) {}
	};

	struct GGUFTensorInfo {
		static constexpr size_t MAX_DIMS = 4;

		uint32_t     						num_dim;
		std::array<uint64_t, MAX_DIMS>     	num_element;
		NumberType   						type;
		uint64_t     						offset;

		void * 								data_ptr;

		GGUFTensorInfo(): num_dim(0), num_element{0}, type(NumberType::FP32),
			offset(0), data_ptr(nullptr) {}
	};

	struct GGUFContext {
	public:
		static constexpr size_t DEFAULT_ALIGNMENT = 32;
		static constexpr auto   GGUF_MAGIC        = GGUFHeader::GGUF_MAGIC;
		static constexpr size_t GGUF_MAGIC_SIZE   = GGUF_MAGIC.size();

	public:
		GGUFHeader                              header;
		std::map<std::string, GGUFValue>        kv_pairs;
		std::map<std::string, GGUFTensorInfo>   infos;
		TensorNameTable         				model_tensor_table;

		std::string                 			arch_name;

		size_t alignment;
		size_t offset;
		size_t size;

	public:
		GGUFContext() : arch_name("unknown"), alignment(DEFAULT_ALIGNMENT), offset(0), size(0) {}
		
		GGUFContext(GGUFContext &&other) noexcept = default;

	public:
		template<class T>
		void add_gguf_value(const std::string &key, T &&value) {
			auto iter_pair = kv_pairs.insert({key, {std::forward<T>(value)}});
			spy_assert(iter_pair.second, "Cannot insert gguf value by key: {}", key);
		}

	public:
		GGUFValue find_gguf_value(const std::string &key) const {
			auto iter = kv_pairs.find(key);
			spy_assert(iter != kv_pairs.end(), "Cannot find gguf value by key: {}", key);
			return iter->second;
		}

		GGUFValue find_gguf_value(const LLMKey key) const {
			const std::string key_name = get_LLM_name(key, arch_name);
			return find_gguf_value(key_name);
		}

		template<class T>
		T find_gguf_value_option(const LLMKey key, const T &default_val) const {
			const std::string key_name = get_LLM_name(key, arch_name);
			const auto option = find_gguf_value_option(key_name);
			if (option.has_value()) { return option->get_value<T>(); }
			return default_val;
		}

		std::optional<GGUFValue> find_gguf_value_option(const std::string &key) const {
			auto iter = kv_pairs.find(key);
			if (iter == kv_pairs.end()) { return std::nullopt; }
			return iter->second;
		}

		std::optional<GGUFValue> find_gguf_value_option(const LLMKey key) const {
			const std::string key_name = get_LLM_name(key, arch_name);
			return find_gguf_value_option(key_name);
		}
	};

	/*!
	 * @brief Metadata of tensor stored in the model file
	 */
	struct ModelTensorMetadata {
		static constexpr size_t MAX_NAME_LEN = 32;

		/* The dimension of tensor */
		/// The name of the tensor
		char tensor_name[MAX_NAME_LEN];
		/// The shape of the tensor
		Shape shape;

		/* The information of tensor stored in the file */
		/// The offset of the tensor in the model file
		size_t offset;
	};

	/*!
	 * @brief The table stored in the file denoting the metadata of all tensors
	 */
	struct ModelTensorMetadataTable {
		/// The number of tensor
		size_t num_tensor;
		/// The array of all tensor metadata
		ModelTensorMetadata metadata_table[];
	};

}// namespace spy