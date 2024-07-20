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
#include "graph/op_node.h"
#include "graph/data_node.h"
#include "graph/graph.h"
#include "backend/config.h"
#include "scheduler/default.h"

namespace spy {

	struct GraphControlHeader {
		friend class GraphView;
	protected:
		std::vector<DataNode *>                     data_node_ptr_array_;
		std::vector<OperatorNode *>                 op_node_ptr_array_;

		std::vector<RelaxedAtomWrapper<int>>            data_recv_count_array_;
		std::vector<RelaxedAtomWrapper<int>>            data_send_count_array_;
		std::vector<RelaxedAtomWrapper<int>>            op_recv_count_array_;

	public:
		void reserve(size_t num_data_node, size_t num_op_node) {
			data_node_ptr_array_.resize(num_data_node, nullptr);
			op_node_ptr_array_.resize(num_op_node, nullptr);
			data_recv_count_array_.resize(num_data_node, 0);
			data_send_count_array_.resize(num_data_node, 0);
			op_recv_count_array_.resize(num_op_node, 0);
		}

		void init_node_id() const {
			for (size_t id = 0; DataNode *data_node_ptr: data_node_ptr_array_) { data_node_ptr->info.id = id++; }
			for (size_t id = 0; OperatorNode *op_node_ptr: op_node_ptr_array_) { op_node_ptr->info.id = id++; }
		}

		void init_dep_count() {
			data_recv_count_array_.resize(data_node_ptr_array_.size(), 0);
			data_send_count_array_.resize(data_node_ptr_array_.size(), 0);
			op_recv_count_array_.resize(op_node_ptr_array_.size(), 0);

			for (const OperatorNode *op_node_ptr: op_node_ptr_array_) {
				op_recv_count_array_[op_node_ptr->info.id].store(op_node_ptr->num_input(), std::memory_order_relaxed);

				for (const DataNode *data_node_ptr: op_node_ptr->info.output) {
					data_recv_count_array_[data_node_ptr->info.id].value.fetch_add(1, std::memory_order_relaxed);
				}
			}
			for (const DataNode *data_node_ptr: data_node_ptr_array_) {
				data_send_count_array_[data_node_ptr->info.id].store(data_node_ptr->num_output(), std::memory_order_relaxed);
			}
		}

	public:
		size_t num_data_node() const { return data_node_ptr_array_.size();  }
		size_t num_op_node()   const { return op_node_ptr_array_.size();    }

		DataNode *    data_node(NodeID id) const { return data_node_ptr_array_[id]; }
		OperatorNode *  op_node(NodeID id) const { return op_node_ptr_array_[id];   }

		RelaxedAtomWrapper<int> &data_recv(NodeID id) { return data_recv_count_array_[id]; }
		RelaxedAtomWrapper<int> &data_send(NodeID id) { return data_send_count_array_[id]; }
		RelaxedAtomWrapper<int> &  op_recv(NodeID id) { return op_recv_count_array_[id];   }

		RelaxedAtomWrapper<int> &data_recv(const DataNode *node_ptr)      { return data_recv_count_array_[node_ptr->info.id]; }
		RelaxedAtomWrapper<int> &data_send(const DataNode *node_ptr)      { return data_send_count_array_[node_ptr->info.id]; }
		RelaxedAtomWrapper<int> &  op_recv(const OperatorNode *node_ptr)  { return op_recv_count_array_[node_ptr->info.id];   }
	};

	class GraphView {
	private:
		std::vector<const Graph *> graph_ptr_array_;

	public:
		GraphView() = default;

		GraphView(const Graph *graph_ptr): GraphView({graph_ptr}) {}

		GraphView(std::initializer_list<const Graph *> &&graph_view_list): graph_ptr_array_(graph_view_list) {}

		~GraphView() = default;

	public:
		GraphView &concat(GraphView &other_view) {
			graph_ptr_array_.insert(graph_ptr_array_.end(),
									other_view.graph_ptr_array_.begin(),
									other_view.graph_ptr_array_.end());
			return *this;
		}

		GraphControlHeader get_control_header() const {
			GraphControlHeader control_header;

			/* Reserve enough space */
			size_t total_data_node_num = 0;
			size_t total_op_node_num   = 0;
			for (const Graph *graph_ptr: graph_ptr_array_) { total_data_node_num += graph_ptr->num_data_node(); }
			for (const Graph *graph_ptr: graph_ptr_array_) { total_op_node_num   += graph_ptr->num_op_node();   }
			control_header.reserve(total_data_node_num, total_op_node_num);

			/* Init header */
			for (size_t cur_node_num = 0; const Graph *graph_ptr: graph_ptr_array_) {
				auto &dst_array = control_header.data_node_ptr_array_;
				auto &src_array = graph_ptr->data_nodes_;
				std::transform(src_array.begin(), src_array.end(), dst_array.begin() + cur_node_num,
							   [](auto &node_ptr){ return std::to_address(node_ptr); });
				cur_node_num += src_array.size();
			}
			// The end node of graph should be merged as an identity
			for (size_t cur_node_num = 0; const Graph *graph_ptr: graph_ptr_array_) {
				auto &dst_array = control_header.op_node_ptr_array_;
				auto &src_array = graph_ptr->op_nodes_;
				std::transform(src_array.begin(), src_array.end(), dst_array.begin() + cur_node_num,
				               [](auto &node_ptr){ return std::to_address(node_ptr); });
				cur_node_num += src_array.size();
			}

			control_header.init_node_id();
			control_header.init_dep_count();
			return control_header;
		}
	};

	using TopoNodeQueue = moodycamel::BlockingConcurrentQueue<OperatorNode *>;

	static void step_op_node(TopoNodeQueue &node_queue, OperatorNode *input_node_ptr, GraphControlHeader &header) {
		input_node_ptr->for_each_output([&](const DataNode *data_node_ptr){
			const size_t cur_data_dep_count = --header.data_recv(data_node_ptr);	
			if (cur_data_dep_count == 0) {
				data_node_ptr->for_each_output([&](OperatorNode *start_op_node_ptr){
					const size_t cur_dep_count = --header.op_recv(start_op_node_ptr);
					if (cur_dep_count == 0) { node_queue.enqueue(start_op_node_ptr); }						
				});
			}
		});
	}

	static void try_allocate_outputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
		if (!is_view(cur_node_ptr->op_type)) {
			cur_node_ptr->for_each_output([&](DataNode *data_node_ptr){
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
			++header.data_send(view_src);
		}
	}

	
	static void try_deallocate_inputs(AbstractBackend *backend_ptr, const OperatorNode *cur_node_ptr, GraphControlHeader &header) {
		cur_node_ptr->for_each_output([&](DataNode *data_node_ptr){
			const DataNodeType node_type = data_node_ptr->node_type;

			if (node_type == DataNodeType::Dynamic) {
				const size_t cur_data_dep_count = --header.data_send(data_node_ptr);
				spy_assert(data_node_ptr->view_src == nullptr);
				if (cur_data_dep_count == 0) {
					Tensor &tensor = data_node_ptr->tensor;

					spy_debug(DebugFlag::Memory, "Release node: {:32} ({})", data_node_ptr->name, tensor.get());

					backend_ptr->dealloc_memory(tensor.get(), tensor.total_size());
					tensor.set_data_ptr(nullptr);
				}
			} else if (node_type == DataNodeType::View) {
				DataNode *src_data_node_ptr              = data_node_ptr->view_src;
				const    DataNodeType src_data_node_type = src_data_node_ptr->node_type;
				Tensor   &src_tensor                     = src_data_node_ptr->tensor;
				// We need to count down the dependency of the source, and release it if needed.
				const size_t cur_data_dep_count = --header.data_send(src_data_node_ptr);
				switch (src_data_node_type) {

				case DataNodeType::Constant:
					break;

				case DataNodeType::ShapeDynamic:
				case DataNodeType::DataDynamic:
				case DataNodeType::Dynamic:
					if (cur_data_dep_count == 0) {
						spy_debug(DebugFlag::Memory, "Release view node: {:27} (0x{})", src_data_node_ptr->property.to_string(), src_tensor.get());

						backend_ptr->dealloc_memory(src_tensor.get(), src_tensor.total_size());
						src_tensor.set_data_ptr(nullptr);
					}	break;

				default:
					spy_assert(false);
				}					
			}
		});
	}

	void DefaultGraphScheduler::execute(Graph &graph) {
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
			OperatorNode *cur_node_ptr = nullptr;
			node_queue.wait_dequeue(cur_node_ptr);

			// We reach the end of the graph
			if (cur_node_ptr == nullptr) [[unlikely]] { break; }

			// Allocate to-be-use output
			try_allocate_outputs(backend_ptr_, cur_node_ptr, graph_control_header);

			// Allocate buffer
			spy_debug(DebugFlag::Execute, "Execute {:8} -> {:32}", 
				cur_node_ptr->op_type, cur_node_ptr->input(0)->name
			);

			backend_ptr_->submit(cur_node_ptr, 
				[&num_unfinished_operator, op_step, backend_ptr = backend_ptr_, cur_node_ptr](){
					op_step(cur_node_ptr, backend_ptr);
				}
			);
		}
	}
    
} // namespace spy