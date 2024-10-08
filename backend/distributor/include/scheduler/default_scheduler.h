#pragma once

#ifndef BACKEND_SCHEDULER_HEADER_MACRO
	#warning "Do not include default_scheduler.h manually, please use scheduler/scheduler.h instead."
#endif // BACKEND_SCHEDULER_HEADER_MACRO


#include <atomic>
#include <cstddef>
#ifdef UNOFFICIAL_CONCURRENTQUEUE
    #include <concurrentqueue/blockingconcurrentqueue.h>
#else 
    #include <blockingconcurrentqueue.h>
#endif

#include "util/wrapper/atomic.h"
#include "graph/op_node.h"
#include "graph/data_node.h"
#include "graph/graph.h"
#include "backend/backend.h"
#include "loader/loader.h"
#include "scheduler/common.h"
#include "scheduler/scheduler.h"

namespace spy {

	using TopoNodeQueue = moodycamel::BlockingConcurrentQueue<OperatorNode *>;

	///
	/// @brief Default scheduler. 
	/// @details 
	/// - **Topology Algorithm**: This scheduler uses a simple topology algorithm which step to the next node with no alive input.
	/// - **Schedule Strategy**: This scheduler give the responsibility of decreasing dependency count to the worker threads,
	///		which means it will distribute all available tasks to the worker backends. On the other hand, if there is no task in the queue,
	///		it will wait until any worker finishing its job, counting down the dependency counter and push a new OperatorNode into the task queue.
	///		Specially, it execute the view node directly if it acquires little operation(task_num).
	/// - **Memory Allocation Strategy**: It gives the whole determination to the backend.
	/// 
    class DefaultGraphScheduler final: public GraphScheduler {
    public:
        DefaultGraphScheduler(Backend *backend_ptr): GraphScheduler(backend_ptr) {}

        ~DefaultGraphScheduler() noexcept override = default;

    public:
        void execute(Graph &graph, GraphControlHeader &control_header) override {
			TopoNodeQueue node_queue;

			size_t num_end_node = 0;

			for (const OperatorNode *entry_point: graph.entry_point_array) {
				entry_point->for_each_output<const DataNode>([&control_header](const DataNode *data_node_ptr){
					auto &data_input_count = control_header.data_input(data_node_ptr);
					--data_input_count;
				});
			}

			for (size_t i = 0; i < control_header.num_data_node(); ++i) {
				if (control_header.data_input(i).load(std::memory_order_relaxed) == 0) { // Input nodes or constant weights
					const auto &data_node = graph.get_node<DataNode>(i);

					data_node.for_each_output<OperatorNode>([&control_header, &node_queue](OperatorNode *op_node_ptr){
						auto &op_recv_count = control_header.op_input(op_node_ptr);
						const auto res_count  = --op_recv_count;

						if (res_count == 0) { node_queue.enqueue(op_node_ptr); }
					});
				}

				if (control_header.data_output(i).load(std::memory_order_relaxed) == 0) { // Output nodes
					num_end_node++;
				}
			}

			const auto op_step = [&] (OperatorNode *cur_node_ptr) {
				// Deallocate outdated input
				try_deallocate_inputs(backend_ptr_, cur_node_ptr, control_header);
				// Step forward
				step_op_node(node_queue, cur_node_ptr, control_header);
			};

			while (num_end_node != 0) {
				OperatorNode *cur_node_ptr = nullptr;

				// Wait for a new task
				node_queue.wait_dequeue(cur_node_ptr);

				// If we reach the end of the graph
				if (cur_node_ptr == nullptr) { --num_end_node; continue; }

				// Pass the operator if it has been deactivated
				if (!cur_node_ptr->active) { op_step(cur_node_ptr); continue; }

				// Fetch acquired tensors if needed
				try_fetch_inputs(backend_ptr_, cur_node_ptr, control_header);

				// Allocate to-be-use output
				try_allocate_outputs(backend_ptr_, cur_node_ptr, control_header);

				// Allocate buffer
				spy_debug(DebugFlag::Execute, "Execute {:8} -> {:32}", 
					cur_node_ptr->op_type, cur_node_ptr->name
				);

				backend_ptr_->submit(cur_node_ptr, 
					[op_step, cur_node_ptr](){
						op_step(cur_node_ptr);
					}
				);
			}
		}

	private:
		inline static void step_op_node(TopoNodeQueue &node_queue, OperatorNode *input_node_ptr, GraphControlHeader &header) {
			input_node_ptr->for_each_output<DataNode>([&](DataNode *data_node_ptr){
				const size_t cur_data_dep_count = --header.data_input(data_node_ptr);	

				if (cur_data_dep_count == 0) {
					if (!input_node_ptr->active && data_node_ptr->all_input_deactivated()) [[unlikely]] {
						data_node_ptr->active = false;
					}
					
					if (data_node_ptr->num_output() == 0) { // Push a flag if reach the end
						node_queue.enqueue(nullptr); 
					} else { // Otherwise, step forward to the next operator nodes.
						data_node_ptr->for_each_output<OperatorNode>([&](OperatorNode *start_op_node_ptr){
							const size_t cur_dep_count = --header.op_input(start_op_node_ptr);
							if (cur_dep_count == 0) { 
								node_queue.enqueue(start_op_node_ptr); 

								if (!data_node_ptr->active && start_op_node_ptr->all_input_deactivated()) [[unlikely]] {
									start_op_node_ptr->active = false;
								}
							}
						});					
					}
				}
			});
		}

		inline static void try_fetch_inputs(Backend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
			cur_node_ptr->for_each_input<DataNode>([&](DataNode *data_node_ptr){
				Tensor &tensor = data_node_ptr->tensor;
				const DataNodeType node_type = data_node_ptr->node_prop.node_type;

				if (node_type == DataNodeType::Constant) {
					const size_t tensor_size = tensor.total_size();

					ModelLoader *loader_ptr = header.loader_ptr_;
					std::span<uint8_t> data_area = loader_ptr->load(data_node_ptr->name);
					tensor.set_data_ptr(data_area.data());

					spy_assert(data_area.size() == tensor_size, 
						"expect the size of the output tensor to be {} (cur: {})", 
						tensor_size, data_area.size()
					);				
				} else if (node_type == DataNodeType::Cache) {
					const size_t tensor_size = tensor.total_size();
					if (data_node_ptr->tensor.get() == nullptr) {
						void *data_ptr = backend_ptr->alloc_memory(tensor_size);
						tensor.set_data_ptr(data_ptr);
					}
				}
			});
		}

		inline static void try_allocate_outputs(Backend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
			if (!is_view(cur_node_ptr->op_type)) {
				cur_node_ptr->for_each_output<DataNode>([&](DataNode *data_node_ptr){
					Tensor &tensor = data_node_ptr->tensor;
					if (tensor.get() == nullptr) {
						void *data_ptr = backend_ptr->alloc_memory(tensor.total_size());
						tensor.set_data_ptr(data_ptr);
					}				
				});
			} else {
				DataNode *input_node_ptr  = cur_node_ptr->input_data(0);
				DataNode *output_node_ptr = cur_node_ptr->output<DataNode>(0);
				DataNode *view_src = (input_node_ptr->view_src == nullptr) ? input_node_ptr : input_node_ptr->view_src;
				output_node_ptr->view_src = view_src;
				// Count up the dependency of the source because we fork a new view
				++header.data_output(view_src);
			}
		}

		inline static void try_deallocate_inputs(Backend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
			cur_node_ptr->for_each_input<DataNode>([&](DataNode *data_node_ptr){
				const DataNodeType node_type = data_node_ptr->node_prop.node_type;

				if (data_node_ptr->is_view()) {
					DataNode *src_data_node_ptr              = data_node_ptr->view_src;
					const    DataNodeType src_data_node_type = src_data_node_ptr->node_prop.node_type;
					Tensor   &src_tensor                     = src_data_node_ptr->tensor;
					// We need to count down the dependency of the source, and release it if needed.
					const size_t cur_data_dep_count = --header.data_output(src_data_node_ptr);
					switch (src_data_node_type) {
					case DataNodeType::Constant:
					case DataNodeType::IO:
					case DataNodeType::Cache:
						break;

					case DataNodeType::ShapeDynamic:
					case DataNodeType::DataDynamic:
					case DataNodeType::Dynamic:
						if (cur_data_dep_count == 0 && src_tensor.get() != nullptr) {
							spy_debug(DebugFlag::Memory, "Release view node: {:27} (0x{})", src_data_node_ptr->to_string(), src_tensor.get());

							backend_ptr->dealloc_memory(src_tensor.get(), src_tensor.total_size());
							src_tensor.set_data_ptr(nullptr);
						}	break;

					default:
						spy_assert(false);
					}					
				} else if (node_type != DataNodeType::Constant && node_type != DataNodeType::IO) {
					Tensor &tensor = data_node_ptr->tensor;
					const size_t cur_data_dep_count = --header.data_output(data_node_ptr);
					spy_assert(data_node_ptr->view_src == nullptr);

					if (cur_data_dep_count == 0 && tensor.get() != nullptr) {
						spy_debug(DebugFlag::Memory, "Release node: {:32} ({})", data_node_ptr->name, tensor.get());

						backend_ptr->dealloc_memory(tensor.get(), tensor.total_size());
						tensor.set_data_ptr(nullptr);
					}
				} 
			});
		}
    };

} // namespace spy
