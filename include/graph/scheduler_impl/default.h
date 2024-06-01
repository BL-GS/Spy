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
		using TopoNodeQueue = moodycamel::BlockingConcurrentQueue<NodeCredit>;
		
	public:
		GraphSchedulerImpl(const std::vector<AbstractBackend *> &backend_array): GraphScheduler(backend_array) {}

		~GraphSchedulerImpl() noexcept override = default;

	public:
		
		void reserve(Graph *graph_ptr) override { 
			// By default, do not reserve any space 
		}

		void execute(Graph *graph_ptr) override {
			using CounterArrayType = std::vector<RelaxedAtomWrapper<size_t>>;
			auto data_recv_counts    = graph_ptr->get_data_recv_count<CounterArrayType>();
			auto data_send_counts    = graph_ptr->get_data_send_count<CounterArrayType>();
			auto op_recv_counts 	 = graph_ptr->get_op_recv_count<CounterArrayType>();

			TopoNodeQueue node_queue;

			for (size_t i = 0; i < data_recv_counts.size(); ++i) {
				if (data_recv_counts[i].load(std::memory_order::relaxed) == 0) {
					const auto &data_node = graph_ptr->get_data_node(i);
					const auto &op_nodes  = data_node->get_output();

					for (const auto &op_node: op_nodes) {
						const auto op_credit  = op_node->credit;
						const auto op_node_id = op_credit.node_id;
						const auto res_count  = --op_recv_counts[op_node_id];

						if (res_count == 0) { node_queue.enqueue(op_credit); }
					}
				}
			}

			const auto op_step = [&] (OperatorNode *cur_node_ptr,  AbstractBackend *backend_ptr) {
				// Deallocate outdated input
				try_deallocate_inputs(backend_ptr, cur_node_ptr, data_send_counts);
				// Step forward
				const NodeCredit cur_credit = cur_node_ptr->credit;

				step_op_node(graph_ptr, node_queue, data_recv_counts, op_recv_counts, cur_credit);
			};

			while (true) {
				// TODO: select backend adaptively
				AbstractBackend *backend_ptr = backend_array_[0];

				NodeCredit cur_credit = Graph::INVALID_NODE_CREDIT;
				node_queue.wait_dequeue(cur_credit);

				// We reach the end of the graph
				if (cur_credit == Graph::OUTPUT_NODE_CREDIT) [[unlikely]] { break; }

				OperatorNode *cur_node_ptr = graph_ptr->get_node_content<OperatorNode>(cur_credit);

				// Allocate to-be-use output
				 try_allocate_outputs(backend_ptr, cur_node_ptr, data_send_counts);

				// Allocate buffer
				spy_debug(DebugFlag::Execute, "Execute {:32} -> {}", cur_node_ptr->name, magic_enum::enum_name(cur_node_ptr->op_type));

				backend_ptr->submit(cur_node_ptr, 
					[op_step, backend_ptr, cur_node_ptr](){ 
						op_step(cur_node_ptr, backend_ptr); 
					} 
				);
			}
		}

	private:
		static void step_op_node(const Graph *graph_ptr, TopoNodeQueue &node_queue,
								 auto &data_dep_count, auto &op_dep_count, NodeCredit cur_node_credit) {
			const OperatorNode *input_node_ptr = graph_ptr->get_node_content<OperatorNode>(cur_node_credit);
			const auto &data_nodes = input_node_ptr->get_output();
			for (const DataNode *data_node_ptr: data_nodes) {
				const NodeCredit data_credit = data_node_ptr->credit;

				const size_t cur_data_dep_count = --data_dep_count[data_credit.node_id];

				if (cur_data_dep_count == 0) {
					const auto &start_op_nodes = data_node_ptr->get_output();
					for (const OperatorNode *start_op_node_ptr: start_op_nodes) {
						const NodeCredit start_node_credit = start_op_node_ptr->credit;

						const size_t cur_dep_count = --op_dep_count[start_node_credit.node_id];
						if (cur_dep_count == 0) { node_queue.enqueue(start_node_credit); }
					}
				}
			}
		}

	private:
		static void try_allocate_outputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, auto &data_dep_count) {
			const auto &inputs = cur_node_ptr->get_input();
			const auto &outputs = cur_node_ptr->get_output();

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
				const NodeCredit src_credit = view_src->credit;
				++data_dep_count[src_credit.node_id];
			}
		}

		
		static void try_deallocate_inputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, auto &data_dep_count) {
			const auto &inputs = cur_node_ptr->get_input();

			for (DataNode *data_node_ptr: inputs) {
				const NodeCredit node_credit = data_node_ptr->credit;

				if (data_node_ptr->data_type == DataNodeType::Variable) {
					const size_t cur_data_dep_count = --data_dep_count[node_credit.node_id];
					spy_assert(data_node_ptr->view_src == nullptr);
					if (cur_data_dep_count == 0) {
						Tensor &tensor = data_node_ptr->tensor;

						spy_debug(DebugFlag::Memory, "Release node: {:32} ({})", data_node_ptr->name, tensor.get());

						backend_ptr->dealloc_memory(tensor.get(), tensor.total_size());
						tensor.set_data_ptr(nullptr);
					}
				} else if (data_node_ptr->data_type == DataNodeType::View) {
					const NodeCredit src_node_credit = data_node_ptr->view_src->credit;
					DataNode *src_data_node_ptr 	 = data_node_ptr->view_src;
					Tensor &src_tensor = src_data_node_ptr->tensor;
					// We need to count down the dependency of the source, and release it if needed.
					const size_t cur_data_dep_count = --data_dep_count[src_node_credit.node_id];
					switch (src_data_node_ptr->data_type) {

					case DataNodeType::Constant:
					case DataNodeType::Buffered:
						break;

					case DataNodeType::Variable:
						if (cur_data_dep_count == 0) {
							spy_debug(DebugFlag::Memory, "Release view node: {:27} (0x{})", src_data_node_ptr->name, src_tensor.get());

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