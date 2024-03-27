#pragma once

#include <atomic>
#include <span>
#include <vector>
#include <concurrentqueue/blockingconcurrentqueue.h>

#include "util/atomic.h"
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
			const auto &origin_back_dep_count = graph_ptr->get_back_dep_count();
			const auto &origin_dep_count = graph_ptr->get_dep_count();	
			std::vector<RelaxedAtomWrapper<size_t>> back_dep_count(origin_back_dep_count.begin(), origin_back_dep_count.end());
			std::vector<RelaxedAtomWrapper<size_t>> dep_count(origin_dep_count.begin(), origin_dep_count.end());

			TopoNodeQueue node_queue;
			step_op_node(graph_ptr, node_queue, dep_count, Graph::INPUT_NODE_CREDIT);

			const auto op_step = [graph_ptr, &node_queue, &dep_count, &back_dep_count] (OperatorNode *cur_node_ptr, const std::span<uint8_t> &buffer, AbstractBackend *backend_ptr) {
				deallocate_buffer(backend_ptr, buffer);
				// Deallocate outdated input
				try_deallocate_inputs(backend_ptr, cur_node_ptr, back_dep_count);
				// Step forward
				const NodeCredit cur_credit = cur_node_ptr->get_credit();
				step_op_node(graph_ptr, node_queue, dep_count, cur_credit);
			};

			while (true) {
				// TODO: select backend adaptively
				AbstractBackend *backend_ptr = backend_array_[0];

				NodeCredit cur_credit = Graph::INVALID_NODE_CREDIT;
				node_queue.wait_dequeue(cur_credit);

				// We reach the end of graph
				if (cur_credit == Graph::OUTPUT_NODE_CREDIT) [[unlikely]] { break; }
				OperatorNode *cur_node_ptr = graph_ptr->get_node_content<OperatorNode>(cur_credit);
				OperatorType op_type = cur_node_ptr->op_type;

				// Allocate to-be-use output
				try_allocate_outputs(backend_ptr, cur_node_ptr, back_dep_count);

				// Allocate buffer
				std::span<uint8_t> buffer_span = allocate_buffer(backend_ptr, cur_node_ptr);

				SPY_DEBUG_FMT_OPTION(Execute, "Execute {:32} -> {}", cur_node_ptr->get_name(), magic_enum::enum_name(cur_node_ptr->op_type));

				const size_t task_num = backend_ptr->get_task_num(cur_node_ptr);
				if (is_view(op_type) && task_num == 1) { // For view operator, which contains little operation, we do not need to bother thread pool.
					const OperatorEnvParam param {
						.concurrency = 1,
						.tid 		 = 0,
						.buffer_span = buffer_span
					};
					backend_ptr->execute(param, cur_node_ptr);
					op_step(cur_node_ptr, buffer_span, backend_ptr);
				} else {
					const size_t max_concurrency 	= backend_ptr->get_max_concurrency();
					const int concurrency 			= std::min(task_num, max_concurrency);

					auto task_counter = std::make_shared<std::atomic_int>(concurrency);
					backend_ptr->submit([cur_node_ptr, backend_ptr, concurrency, buffer_span, task_counter = std::move(task_counter), &op_step](int tid){
						const OperatorEnvParam param {
							.concurrency = concurrency,
							.tid		 = tid,
							.buffer_span = buffer_span
						};
						backend_ptr->execute(param, cur_node_ptr);

						int task_id = task_counter->fetch_sub(1);

						if (task_id == 1) {
							op_step(cur_node_ptr, buffer_span, backend_ptr);
						}
					}, concurrency);
				}
			}
		}

	private:
		static void step_op_node(const Graph *graph_ptr, TopoNodeQueue &node_queue,
								 auto &dep_count, NodeCredit cur_node_credit) {
			const OperatorNode *input_node_ptr = graph_ptr->get_node_content<OperatorNode>(cur_node_credit);
			const auto &data_nodes = input_node_ptr->get_output();
			for (const BaseNode *data_node_ptr: data_nodes) {
				const NodeCredit data_credit = data_node_ptr->get_credit();

				const size_t cur_data_dep_count = --dep_count[data_credit];

				if (cur_data_dep_count == 0) {
					const auto &start_op_nodes = data_node_ptr->get_output();
					for (const BaseNode *start_op_node_ptr: start_op_nodes) {
						const NodeCredit start_node_credit = start_op_node_ptr->get_credit();

						const size_t cur_dep_count = --dep_count[start_node_credit];
						if (cur_dep_count == 0) { node_queue.enqueue(start_node_credit); }
					}
				}
			}
		}

	private:
		static void try_allocate_outputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, auto &back_dep_count) {
			const auto &inputs = cur_node_ptr->get_input();
			const auto &outputs = cur_node_ptr->get_output();

			if (!is_view(cur_node_ptr->op_type)) {
				for (BaseNode *node_ptr: outputs) {
					DataNode *data_node_ptr = static_cast<DataNode *>(node_ptr);
					Tensor &tensor = data_node_ptr->tensor;
					if (tensor.get() == nullptr) {
						void *data_ptr = backend_ptr->alloc_memory(tensor.total_size());
						tensor.set_data_ptr(data_ptr);
					}
				}
			} else {
				DataNode *input_node_ptr  = static_cast<DataNode *>(inputs[0]);
				DataNode *output_node_ptr = static_cast<DataNode *>(outputs[0]);
				if (input_node_ptr->view_src == nullptr) {
					output_node_ptr->view_src = input_node_ptr;
				} else {
					output_node_ptr->view_src = input_node_ptr->view_src;
				}
				// Count up the dependency of the source because we fork a new view
				NodeCredit src_credit = output_node_ptr->view_src->get_credit();
				++back_dep_count[src_credit];
			}
		}

		
		static void try_deallocate_inputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, auto &back_dep_count) {
			const auto &inputs = cur_node_ptr->get_input();

			for (BaseNode *node_ptr: inputs) {
				const NodeCredit node_credit = node_ptr->get_credit();
				DataNode *data_node_ptr 	 = static_cast<DataNode *>(node_ptr);

				if (data_node_ptr->data_type == DataNodeType::Variable) {
					const size_t cur_data_back_dep_count = --back_dep_count[node_credit];
					SPY_ASSERT(data_node_ptr->view_src == nullptr);
					if (cur_data_back_dep_count == 0) { 
						Tensor &tensor = data_node_ptr->tensor;
						backend_ptr->dealloc_memory(tensor.get(), tensor.total_size());
						tensor.set_data_ptr(nullptr);

						SPY_DEBUG_FMT_OPTION(Memory, "Release node: {:32} (0x{})", data_node_ptr->get_name(), tensor.get());
					}
				} else if (data_node_ptr->data_type == DataNodeType::View) {
					const NodeCredit src_node_credit = data_node_ptr->view_src->get_credit();
					DataNode *src_data_node_ptr 	 = static_cast<DataNode *>(data_node_ptr->view_src);
					Tensor &src_tensor = src_data_node_ptr->tensor;
					// We need to count down the dependency of the source, and release it if needed.
					const size_t cur_data_back_dep_count = --back_dep_count[src_node_credit];
					switch (src_data_node_ptr->data_type) {

					case DataNodeType::Constant:
					case DataNodeType::Buffered:
						break;

					case DataNodeType::Variable:
						if (cur_data_back_dep_count == 0) {
							backend_ptr->dealloc_memory(src_tensor.get(), src_tensor.total_size());
							src_tensor.set_data_ptr(nullptr);

							SPY_DEBUG_FMT_OPTION(Memory, "Release view node: {:27} (0x{})", src_data_node_ptr->get_name(), src_tensor.get());
						}	break;

					default:
						SPY_ASSERT(false);
					}					
				}
			}
		}

		static std::span<uint8_t> allocate_buffer(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr) {
			const size_t buffer_size = backend_ptr->get_buffer_size(cur_node_ptr);
			// no buffer
			if (buffer_size == 0) { return {}; }
			// allocate buffer
			return { static_cast<uint8_t *>(backend_ptr->alloc_memory(buffer_size)), buffer_size };
		}

		static void deallocate_buffer(AbstractBackend *backend_ptr, const std::span<uint8_t> &buffer) {
			backend_ptr->dealloc_memory(buffer.data(), buffer.size());
		}
	};

    using DefaultGraphScheduler = GraphSchedulerImpl<SchedulerType::Default>;
    
} // namespace spy