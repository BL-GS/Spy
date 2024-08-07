#pragma once

#include "operator/type.h"
#include "backend/type.h"
#include "backend/config.h"

namespace spy::cpu {

	struct OperatorResult;
	struct OperatorEnvParam;
	struct ControlHeader;

    class CPUBackend: public Backend {
	public:
		using TaskFunc = OperatorResult (*)(CPUBackend *, const OperatorEnvParam &, OperatorNode *);
		
	public:
		CPUBackend(): Backend(BackendType::Host) {}

		~CPUBackend() noexcept override = default;

	public: /* Memory Operation */
		size_t get_max_memory_capacity() 	const override;

		size_t get_avail_memory_capacity() 	const override;

	public: /* Data Management */
		void *alloc_memory(size_t size) override { return new uint8_t[size]; }
		void  dealloc_memory(void *ptr, [[maybe_unused]]size_t size) 	override { delete[] static_cast<uint8_t *>(ptr); }

	public: /* Processor Operation */
		size_t get_max_concurrency() 	const override;
        
		size_t get_avail_concurrency() 	const override;

	public:
		bool is_support(OperatorType op_type) const override;

		/*!
		 * @brief Execute the operator
		 * @param param The parameter of environment. Specifically, it denote the concurrency and thread id of CPU backend.
		 * @param op_node The OperatorNode deriving from OperatorDefinition<T_op_type>, which store necessary operands and hyper parameters.
		 * @return true if supported, otherwise false.
		 */
		virtual OperatorResult execute(const OperatorEnvParam &param, OperatorNode *node_ptr);

	protected:
		TaskFunc get_execute_func(OperatorType op_type) const;

		std::shared_ptr<ControlHeader> get_control_header(OperatorType op_type, OperatorNode *node_ptr);
	};

}