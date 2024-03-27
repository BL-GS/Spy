#pragma once

#include <memory>
#include <magic_enum_switch.hpp>

#include "graph/type.h"
#include "graph/config.h"
#include "graph/scheduler_impl/default.h"

namespace spy {

	using magic_enum::enum_switch;

	/// 
	/// @brief A factory class scheduling graph scheduler dynamically or staically.
	/// @note Please look up the `GraphSchedulerImpl<SchedulerType>` for the detail of scheduler.
	/// 
	class GraphSchedulerBuilder {
	public:
		/*!
		 * @brief Distribute the scheduler dynamically
		 */
		static std::unique_ptr<GraphScheduler> build_scheduler(SchedulerType scheduler_type, const std::vector<AbstractBackend *> &backend_array) {
			return enum_switch([&backend_array](const auto T_scheduler_type){
				return static_cast<std::unique_ptr<GraphScheduler>>(std::make_unique<GraphSchedulerImpl<T_scheduler_type>>(backend_array));
			}, scheduler_type);
		}

		/*!
		 * @brief Distribute the scheduler statically
		 */
		template<SchedulerType T_scheduler_type>
		static auto build_scheduler(const std::vector<AbstractBackend *> &backend_array) {
			return GraphSchedulerImpl<T_scheduler_type>(backend_array);
		}
	};
    
} // namespace spy