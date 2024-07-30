#include "util/shell/logger.h"
#include "util/wrapper/atomic.h"
#include "graph/graph.h"
#include "backend/config.h"
#include "distributor/policy/simple.h"

namespace spy {

    bool SimpleGraphDistributor::add_backend(AbstractBackend *backend_ptr, const std::string_view sche_policy) {
        if (scheduler_ptr_ != nullptr) {
            spy_warn("trying to overwrite the backend, only the last one will be used");
        }

        scheduler_ptr_ = GraphSchedulerBuilder::build_scheduler(sche_policy, backend_ptr);
        return true;
    }

    void SimpleGraphDistributor::prepare_graph(Graph *graph_ptr) {
        if (graph_ptr_ != nullptr) {
            spy_warn("trying to overwrite the graph");
        }
        graph_ptr_ = graph_ptr;
    }

    void SimpleGraphDistributor::execute() {
        GraphStorage &graph_storage = *graph_ptr_->storage_ptr;

        /* Set up dependency count array */

        const size_t num_data_node = graph_storage.num_data_node();
        const size_t num_op_node   = graph_storage.num_op_node();

        std::vector<RelaxedAtomWrapper<int>> data_input_array(num_data_node, 0);
        std::vector<RelaxedAtomWrapper<int>> data_output_array(num_data_node, 0);
        std::vector<RelaxedAtomWrapper<int>> op_input_array(num_op_node, 0);

        // Denote whether the data is available
        graph_storage.get_data_input_count(data_input_array.begin(), data_input_array.end());
        // Denote whether the data is out of use
        graph_storage.get_data_output_count(data_output_array.begin(), data_output_array.end());
        // Denote whether the operator is available
        graph_storage.get_op_input_count(op_input_array.begin(), op_input_array.end());
        
        /* Distribute the graph to the backend and execute */

        // Simple distributor does not divide graph. 
        // Therefore, just give it to the graph and launch the backend
        GraphControlHeader control_header {
            .data_input_array_  = data_input_array,
            .data_output_array_ = data_output_array,
            .op_input_array_    = op_input_array,

            .loader_ptr_ = loader_ptr_
        };

        scheduler_ptr_->execute(*graph_ptr_, control_header);
    }

} // namespace spy