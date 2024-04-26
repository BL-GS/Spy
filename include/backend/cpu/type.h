/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <magic_enum_switch.hpp>

#include "backend/config.h"
#include "backend/cpu/operator_impl.h"

namespace spy {

	class CPUBackend: public AbstractBackend {
	public:
		CPUBackend() = default;

		~CPUBackend() noexcept override = default;

	public: /* Memory Operation */
		size_t get_max_memory_capacity() 	const override;
		size_t get_avail_memory_capacity() 	const override;

	public: /* Data Management */
		void *alloc_memory(size_t size) override { return new uint8_t[size];             }
		void  dealloc_memory(void *ptr, [[maybe_unused]]size_t size) 	override { delete[] static_cast<uint8_t *>(ptr); }

	public: /* Processor Operation */
		size_t get_max_concurrency() 	const override;
		size_t get_avail_concurrency() 	const override;

	public:
		size_t get_task_num(const OperatorNode *node_ptr) const override {
			const OperatorType op_type = node_ptr->op_type;

			auto func = magic_enum::enum_switch([](auto type){
				return cpu::OperatorImpl<type>::get_task_num;
			}, op_type);
			return func(this, node_ptr);
		}

		size_t get_buffer_size(const OperatorNode *node_ptr) const override {
			const OperatorType op_type = node_ptr->op_type;

			auto func = magic_enum::enum_switch([](auto type){
				return cpu::OperatorImpl<type>::get_buffer_size;
			}, op_type);
			return func(this, node_ptr);
		}

		bool execute(const OperatorEnvParam &param, OperatorNode *node_ptr) override {
			const OperatorType op_type = node_ptr->op_type;

			auto func = magic_enum::enum_switch([](auto type){
				return cpu::OperatorImpl<type>::execute;
			}, op_type);
			return func(this, param, node_ptr);
		}
	};
}  // namespace spy
