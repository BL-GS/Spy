#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <span>
#include <vector>
#ifdef UNOFFICIAL_CONCURRENTQUEUE
    #include <concurrentqueue/blockingconcurrentqueue.h>
#else 
    #include <blockingconcurrentqueue.h>
#endif

#include "util/wrapper/atomic.h"
#include "graph/type.h"
#include "graph/config.h"
#include "graph/graph.h"
#include "operator/config.h"

namespace spy {

	///
	/// @brief Default scheduler. 
	/// @details 
	/// - **Topology Algorithm**: This scheduler uses a simple topology algorithm which step to the next node with no alive input.
	/// - **Schedule Strategy**: This scheduler give the responsibility of decresing dependency count to the worker threads, 
	///		which means it will distribute all avaiable tasks to the worker backends. On the other hand, if there is no task in the queue,
	///		it will wait until any worker finishing its job, counting down the dependency counter and push a new OperatorNode into the task queue.
	///		Specially, it execute the view node directly if it acquires little operation(task_num).
	/// - **Memory Allocation Strategy**: It gives the whole determination to the backend.
	/// 
    template<>
	class GraphSchedulerImpl<SchedulerType::Default> final: public GraphScheduler {
	public:
		using TopoNodeQueue = moodycamel::BlockingConcurrentQueue<OperatorNode *>;
		
	public:
		GraphSchedulerImpl(const std::vector<AbstractBackend *> &backend_array): GraphScheduler(backend_array) {}

		~GraphSchedulerImpl() noexcept override = default;

	public:
		void execute(GraphView &graph_view) override {
			GraphControlHeader graph_control_header = graph_view.get_control_header();
			TopoNodeQueue node_queue;

			std::atomic<size_t> num_unfinished_operator{ graph_control_header.num_op_node() };

			for (size_t i = 0; i < graph_control_header.num_data_node(); ++i) {
				if (graph_control_header.data_recv(i).load(std::memory_order_relaxed) == 0) {
					const auto &data_node = graph_control_header.data_node(i);
					const auto &op_nodes   = data_node->output();

					for (auto *op_node_ptr: op_nodes) {
						auto &op_recv_count = graph_control_header.op_recv(op_node_ptr);
						const auto res_count  = --op_recv_count;

						if (res_count == 0) { node_queue.enqueue(op_node_ptr); }
					}
				}
			}

			const auto op_step = [&] (OperatorNode *cur_node_ptr,  AbstractBackend *backend_ptr) {
				// Deallocate outdated input
				try_deallocate_inputs(backend_ptr, cur_node_ptr, graph_control_header);
				// Step forward
				step_op_node(node_queue, cur_node_ptr, graph_control_header);
				// Push nullptr if reach the end
				const size_t num_left_op = --num_unfinished_operator;
				if (num_left_op == 0) [[unlikely]] { node_queue.enqueue(nullptr); }
			};

			while (true) {
				// TODO: select backend adaptively
				AbstractBackend *backend_ptr = backend_array_[0];

				OperatorNode *cur_node_ptr = nullptr;
				node_queue.wait_dequeue(cur_node_ptr);

				// We reach the end of the graph
				if (cur_node_ptr == nullptr) [[unlikely]] { break; }

				// Allocate to-be-use output
				try_allocate_outputs(backend_ptr, cur_node_ptr, graph_control_header);

				// Allocate buffer
				spy_debug(DebugFlag::Execute, "Execute {:8} -> {:32}", 
					cur_node_ptr->op_type,
					cur_node_ptr->input(0).property.to_string()
				);

				backend_ptr->submit(cur_node_ptr, 
					[&num_unfinished_operator, op_step, backend_ptr, cur_node_ptr](){
						op_step(cur_node_ptr, backend_ptr);
					}
				);
			}
		}

	private:
		static void step_op_node(TopoNodeQueue &node_queue,
								 OperatorNode *input_node_ptr, GraphControlHeader &header) {
			const auto &data_nodes = input_node_ptr->output();
			for (const DataNode *data_node_ptr: data_nodes) {
				const size_t cur_data_dep_count = --header.data_recv(data_node_ptr);

				if (cur_data_dep_count == 0) {
					const auto &start_op_nodes = data_node_ptr->output();
					for (OperatorNode *start_op_node_ptr: start_op_nodes) {
						const size_t cur_dep_count = --header.op_recv(start_op_node_ptr);
						if (cur_dep_count == 0) { node_queue.enqueue(start_op_node_ptr); }
					}
				}
			}
		}

	private:
		static void try_allocate_outputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
			const auto &inputs = cur_node_ptr->input();
			const auto &outputs = cur_node_ptr->output();

			if (!is_view(cur_node_ptr->op_type)) {
				for (DataNode *data_node_ptr: outputs) {
					Tensor &tensor = data_node_ptr->tensor;
					if (tensor.get() == nullptr) {
						void *data_ptr = backend_ptr->alloc_memory(tensor.total_size());
						tensor.set_data_ptr(data_ptr);
					}
				}
			} else {
				DataNode *input_node_ptr  = inputs[0];
				DataNode *output_node_ptr = outputs[0];
				DataNode *view_src = (input_node_ptr->view_src == nullptr) ? input_node_ptr : input_node_ptr->view_src;
				output_node_ptr->view_src = view_src;
				// Count up the dependency of the source because we fork a new view
				++header.data_send(view_src);
			}
		}

		
		static void try_deallocate_inputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
			const auto &inputs = cur_node_ptr->input();

			for (DataNode *data_node_ptr: inputs) {
				const DataNodeType	node_type	= data_node_ptr->property.node_type;

				if (node_type == DataNodeType::Variable) {
					const size_t cur_data_dep_count = --header.data_send(data_node_ptr);
					spy_assert(data_node_ptr->view_src == nullptr);
					if (cur_data_dep_count == 0) {
						Tensor &tensor = data_node_ptr->tensor;

						spy_debug(DebugFlag::Memory, "Release node: {:32} ({})", data_node_ptr->property.to_string(), tensor.get());

						backend_ptr->dealloc_memory(tensor.get(), tensor.total_size());
						tensor.set_data_ptr(nullptr);
					}
				} else if (node_type == DataNodeType::View) {
					DataNode *src_data_node_ptr              = data_node_ptr->view_src;
					const    DataNodeType src_data_node_type = src_data_node_ptr->property.node_type;
					Tensor   &src_tensor                     = src_data_node_ptr->tensor;
					// We need to count down the dependency of the source, and release it if needed.
					const size_t cur_data_dep_count = --header.data_send(src_data_node_ptr);
					switch (src_data_node_type) {

					case DataNodeType::Constant:
					case DataNodeType::Buffered:
						break;

					case DataNodeType::Variable:
						if (cur_data_dep_count == 0) {
							spy_debug(DebugFlag::Memory, "Release view node: {:27} (0x{})", src_data_node_ptr->property.to_string(), src_tensor.get());

							backend_ptr->dealloc_memory(src_tensor.get(), src_tensor.total_size());
							src_tensor.set_data_ptr(nullptr);
						}	break;

					default:
						spy_assert(false);
					}					
				}
			}
		}
	};

    using DefaultGraphScheduler = GraphSchedulerImpl<SchedulerType::Default>;
    
} // namespace spy