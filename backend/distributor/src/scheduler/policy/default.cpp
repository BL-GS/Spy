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
#include "backend/config.h"
#include "scheduler/common.h"
#include "scheduler/policy/default.h"

namespace spy {

	using TopoNodeQueue = moodycamel::BlockingConcurrentQueue<OperatorNode *>;

	static void step_op_node(TopoNodeQueue &node_queue, OperatorNode *input_node_ptr, GraphControlHeader &header) {
		input_node_ptr->for_each_output<DataNode>([&](const DataNode *data_node_ptr){
			const size_t cur_data_dep_count = --header.data_input(data_node_ptr);	
			if (cur_data_dep_count == 0) {
				data_node_ptr->for_each_output<OperatorNode>([&](OperatorNode *start_op_node_ptr){
					const size_t cur_dep_count = --header.op_input(start_op_node_ptr);
					if (cur_dep_count == 0) { node_queue.enqueue(start_op_node_ptr); }						
				});
			}
		});
	}

	static void try_allocate_outputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
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

	
	static void try_deallocate_inputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
		cur_node_ptr->for_each_output<DataNode>([&](DataNode *data_node_ptr){
			const DataNodeType node_type = data_node_ptr->node_prop.node_type;

			if (data_node_ptr->is_view()) {
				DataNode *src_data_node_ptr              = data_node_ptr->view_src;
				const    DataNodeType src_data_node_type = src_data_node_ptr->node_prop.node_type;
				Tensor   &src_tensor                     = src_data_node_ptr->tensor;
				// We need to count down the dependency of the source, and release it if needed.
				const size_t cur_data_dep_count = --header.data_output(src_data_node_ptr);
				switch (src_data_node_type) {

				case DataNodeType::Constant:
					break;

				case DataNodeType::ShapeDynamic:
				case DataNodeType::DataDynamic:
				case DataNodeType::Dynamic:
					if (cur_data_dep_count == 0) {
						spy_debug(DebugFlag::Memory, "Release view node: {:27} (0x{})", src_data_node_ptr->to_string(), src_tensor.get());

						backend_ptr->dealloc_memory(src_tensor.get(), src_tensor.total_size());
						src_tensor.set_data_ptr(nullptr);
					}	break;

				default:
					spy_assert(false);
				}					
			} else if (node_type == DataNodeType::Dynamic) {
				const size_t cur_data_dep_count = --header.data_output(data_node_ptr);
				spy_assert(data_node_ptr->view_src == nullptr);
				if (cur_data_dep_count == 0) {
					Tensor &tensor = data_node_ptr->tensor;

					spy_debug(DebugFlag::Memory, "Release node: {:32} ({})", data_node_ptr->name, tensor.get());

					backend_ptr->dealloc_memory(tensor.get(), tensor.total_size());
					tensor.set_data_ptr(nullptr);
				}
			} 
		});
	}

	void DefaultGraphScheduler::execute(Graph &graph, GraphControlHeader &control_header) {
		TopoNodeQueue node_queue;

		for (size_t i = 0; i < control_header.num_data_node(); ++i) {
			if (control_header.data_input(i).load(std::memory_order_relaxed) == 0) {
				const auto &data_node = graph.get_node<DataNode>(i);

				data_node.for_each_output<OperatorNode>([&control_header, &node_queue](OperatorNode *op_node_ptr){
					auto &op_recv_count = control_header.op_input(op_node_ptr);
					const auto res_count  = --op_recv_count;

					if (res_count == 0) { node_queue.enqueue(op_node_ptr); }
				});
			}
		}

		const auto op_step = [&] (OperatorNode *cur_node_ptr,  AbstractBackend *backend_ptr) {
			// Deallocate outdated input
			try_deallocate_inputs(backend_ptr, cur_node_ptr, control_header);
			// Step forward
			step_op_node(node_queue, cur_node_ptr, control_header);
		};

		while (true) {
			OperatorNode *cur_node_ptr = nullptr;
			node_queue.wait_dequeue(cur_node_ptr);

			// We reach the end of the graph
			if (cur_node_ptr == nullptr) [[unlikely]] { break; }

			// Allocate to-be-use output
			try_allocate_outputs(backend_ptr_, cur_node_ptr, control_header);

			// Allocate buffer
			spy_debug(DebugFlag::Execute, "Execute {:8} -> {:32}", 
				cur_node_ptr->op_type, cur_node_ptr->input(0)->name
			);

			backend_ptr_->submit(cur_node_ptr, 
				[op_step, backend_ptr = backend_ptr_, cur_node_ptr](){
					op_step(cur_node_ptr, backend_ptr);
				}
			);
		}
	}
    
} // namespace spy