/*
 * @author: BL-GS 
 * @date:   2024/3/23
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <array>
#include <numeric>
#include <string>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "util/type/printable.h"
#include "number/number.h"

namespace spy {

	constexpr uint32_t MAX_DIM = 4;

	struct Shape {
	public:
		using DimensionArray = std::array<int64_t, MAX_DIM>;

	public:
		/// The type of underlying number type
		NumberType 	   number_type;
		/// The dimension of tensor
		uint32_t       dim;
		/// The number of slices in each dimension.
		DimensionArray elements;
		/// The accumulated size in each dimension.
		DimensionArray bytes;

	public:
		constexpr Shape(): number_type(NumberType::FP32), dim(0), elements{0}, bytes{0} {}

		/*!
		 * @brief By default, we generate a shape for contiguous structure
		 */
		Shape(uint32_t dim, const DimensionArray &num_element, const NumberType number_type):
				number_type(number_type), dim(dim), elements(num_element), bytes{0} {
			spy_assert(dim <= MAX_DIM, "The dimension should be within the range (0, {}] (cur: {})", MAX_DIM, dim);

			const size_t type_size = get_type_size(number_type);
			// Expect the high dimension to be 1
			std::fill(elements.begin() + dim, elements.end(), 1);
			// Assign the accumulated size of tensor
			bytes[0] = type_size;
			size_t acc_bytes = get_row_size(number_type, elements[0]);
			for (uint32_t i = 1; i < MAX_DIM; ++i) {
				bytes[i]    = acc_bytes;
				acc_bytes  *= elements[i];
			}
		}

		/*!
		 * @brief By default, we generate a shape for contiguous structure
		 */
		Shape(const std::initializer_list<int64_t> &num_element, const NumberType number_type):
				number_type(number_type), dim(num_element.size()), elements{0}, bytes{0} {
			spy_assert(dim <= MAX_DIM, "The dimension should be within the range (0, {}] (cur: {})", MAX_DIM, dim);

			std::copy(num_element.begin(), num_element.end(), elements.begin());
			for (size_t i = dim; i < MAX_DIM; ++i) {
				// Expect the high dimension to be 1
				elements[i] = 1;
			}

			const size_t type_size = get_type_size(number_type);
			size_t acc_byte = get_row_size(number_type, elements[0]);
			bytes[0] = type_size;
			for (uint32_t i = 1; i < MAX_DIM; ++i) {
				// Assign the accumulated size of tensor
				bytes[i]    = acc_byte;
				acc_byte   *= elements[i];
			}
		}

		Shape(uint32_t dim, const DimensionArray &num_element, const DimensionArray &num_byte, const NumberType number_type):
				number_type(number_type), dim(dim), elements(num_element), bytes(num_byte) {
			spy_assert(dim <= MAX_DIM, "The dimension should be within the range (0, {}] (cur: {})", MAX_DIM, dim);
			// Expect the high dimension to be 1
			std::fill(elements.begin() + dim, elements.end(), 1);
			std::fill(bytes.begin() + dim, bytes.end(), elements[dim - 1] * num_byte[dim - 1]);
		}

		Shape(const std::initializer_list<int64_t> &num_element, const std::initializer_list<size_t> &num_byte, const NumberType number_type):
				number_type(number_type), dim(num_element.size()), elements{0}, bytes{0} {
			spy_assert(dim <= MAX_DIM, "The dimension should be within the range (0, {}] (cur: {})", MAX_DIM, dim);

			std::copy(num_element.begin(), num_element.end(), elements.begin());
			std::copy(num_byte.begin(), num_byte.end(), bytes.begin());

			const size_t total_size = elements[dim - 1] * bytes[dim - 1];
			for (uint32_t i = dim; i < MAX_DIM; ++i) {
				elements[i] = 1;
				bytes[i]    = total_size;
			}
		}

		Shape(const Shape &) = default;

		Shape &operator =(const Shape &other) = default;

		bool operator ==(const Shape &other) const {
			return number_type == other.number_type && dim == other.dim &&
				elements == other.elements && bytes == other.bytes;
		};

	public:
		void permute(const DimensionArray &axis) {
			DimensionArray new_elements;
			DimensionArray new_bytes;

			for (uint32_t i = 0; i < MAX_DIM; ++i) { new_elements[i] 	= elements[axis[i]]; }
			for (uint32_t i = 0; i < MAX_DIM; ++i) { new_bytes[i] 	= bytes[axis[i]]; 	 }

			dim 	 = 4;
			elements = new_elements;
			bytes    = new_bytes;
		}

		void transpose() {
			std::swap(elements[0], elements[1]);
			std::swap(bytes[0], bytes[1]);
		}

	public:
		/*!
		 * @brief Get the total size of tensor
		 */
		size_t total_size() 	const { 
			const size_t block_size = get_block_size(number_type);
			size_t size = 0;
			size = bytes[0] * elements[0] / block_size;
			for (uint32_t i = 1; i < dim; ++i) { size += bytes[i] * (elements[i] - 1); }
			return size;
		}

		/*!
		 * @brief Get the total size of tensor
		 */
		size_t row_size() 		const { return get_type_size(number_type) * elements[0] / get_block_size(number_type); }

		/*!
		 * @brief Get the total number of elements
		 */
		size_t total_element() 	const { return std::accumulate(elements.begin(), elements.end(), 1, std::multiplies<size_t>()); }

		/*!
		 * @brief Get the accumulated number of elements in the specific dimension
		 */
		size_t acc_element(size_t acc_dim)	const { return std::accumulate(elements.begin(), elements.begin() + acc_dim, 1, std::multiplies<size_t>()); }

		/*
		 * @brief Get the total number of element in the sub-tensor
		 */
		size_t num_sub_tensor(size_t sub_dim) const { return std::accumulate(elements.begin() + sub_dim, elements.end(), 1, std::multiplies<size_t>()); }

		/*!
		 * @brief Get the number of rows in tensor
		 */
		size_t num_row() const { return num_sub_tensor(1); }

		/*!
		 * @brief Get the total number of block
		 */
		size_t total_block() const { return total_element() / get_block_size(number_type); }

	public: /* View */
		/*!
		 * @brief Whether the shape denote a contiguous space
		 */
		bool is_continuous() const {
			const bool contiguous_0 = (bytes[0] == get_type_size(number_type));
			const bool contiguous_1 = (bytes[1] == bytes[0] * elements[0] / get_block_size(number_type));
			const bool contiguous_2 = (bytes[2] == bytes[1] * elements[1]);
			const bool contiguous_3 = (bytes[3] == bytes[2] * elements[2]);
			return contiguous_0 && contiguous_1 && contiguous_2 && contiguous_3;
		}

		/*!
		 * @brief Whether the shape denote a contiguous space
		 */
		bool is_transposed() const { return bytes[0] > bytes[1]; }

		/*!
		 * @brief Whether the shape denote a contiguous space
		 */
		bool is_permuted() const { return bytes[0] > bytes[1] || bytes[1] > bytes[2] || bytes[2] > bytes[3]; }

	public: /* Utility */
		static bool can_repeat(const Shape &multi, const Shape &single) {
			bool can_repeat = true;
			for (uint32_t i = 0; i < MAX_DIM; ++i) { can_repeat &= (multi.elements[i] % single.elements[i] == 0); }
			return can_repeat;
		}

		std::string to_string() const;

		std::map<std::string_view, std::string> property() const;
	};

	struct Tensor {
	public:
		using DimensionArray = Shape::DimensionArray;

	public:
		/// The shape of tensor
		Shape   	shape;
		/// The pointer to the start address of data
		void *  	data_ptr;

	public:
		Tensor(): data_ptr(nullptr) {}

		Tensor(const Shape &shape, void *data_ptr): shape(shape), data_ptr(data_ptr) { }

		Tensor(const Tensor &other) = default;

	public:
		template<class T = void>
		T *get() const { return static_cast<T *>(data_ptr); }

		template<class T = void>
		T *get(DimensionArray index_array) const {
			// align with data block
			index_array[0] /= get_block_size(type());
	
			DimensionArray offset_array;
			std::transform( index_array.begin(), index_array.end(), shape.bytes.begin(), 
				offset_array.begin(), std::multiplies<size_t>()
			);
			const size_t offset = std::reduce(offset_array.begin(), offset_array.end());
			void *res_ptr = static_cast<uint8_t *>(data_ptr) + offset;
			return static_cast<T *>(res_ptr);
		}

		void set_data_ptr(void *new_data_ptr) { this->data_ptr = new_data_ptr; }

	public: /* Shape Information */
		const Shape &	get_shape()       const { return shape; 					}
		NumberType  	type() 			  const { return shape.number_type; 		}
		size_t      	dim()         	  const { return shape.dim; 				}
		DimensionArray  elements()	  	  const { return shape.elements; 			}
		DimensionArray  bytes()	  		  const { return shape.bytes; 				}
		size_t			total_size()	  const { return shape.total_size(); 		}
		size_t			total_element()	  const { return shape.total_element(); 	}
		size_t          total_block()     const { return shape.total_block(); 		}

	public: /* View Information */
		bool			is_continuous()   const { return shape.is_continuous();   }
		bool			is_transposed()   const { return shape.is_transposed();	}
		bool			is_permuted()	  const { return shape.is_permuted();		}

	public:
		std::map<std::string_view, std::string> property() const;
	};

}  // namespace spy

SPY_PRINTABLE_FORMATTER(spy::Shape);