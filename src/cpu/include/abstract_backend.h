#pragma once

#include "backend/config.h"

namespace spy::cpu {

    class CPUBackend: public AbstractBackend {
	public:
		CPUBackend() = default;

		~CPUBackend() noexcept override = default;

	public: /* Memory Operation */
		size_t get_max_memory_capacity() 	const override;

		size_t get_avail_memory_capacity() 	const override;

	public: /* Data Management */
		void *alloc_memory(size_t size) override { return new uint8_t[size];    
                 }
		void  dealloc_memory(void *ptr, [[maybe_unused]]size_t size) 	override { delete[] static_cast<uint8_t *>(ptr); }

	public: /* Processor Operation */
		size_t get_max_concurrency() 	const override;
        
		size_t get_avail_concurrency() 	const override;

	public:
		size_t get_task_num(const OperatorNode *node_ptr) const override;

		size_t get_buffer_size(const OperatorNode *node_ptr) const override;

		OperatorStatus execute(const OperatorEnvParam &param, OperatorNode *node_ptr) override;
	};

}