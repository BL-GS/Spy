#include <string>
#include <string_view>
#include <map>

#include "graph/graph.h"

namespace spy {

    void Graph::propagate() const {

        auto data_input_count_array = storage_ptr->get_data_input_count();
        auto op_input_count_array   = storage_ptr->get_op_input_count();

        std::queue<OperatorNode *> op_queue;

        for (OperatorNode *entry_point: entry_point_array) {
            entry_point->propagate();
            entry_point->for_each_output([&](BasicNode *input_node_ptr){
                const NodeID node_id = input_node_ptr->id;
                data_input_count_array[node_id]--;
            });
        }

        for (size_t node_id = 0; node_id < data_input_count_array.size(); ++node_id) {
            if (data_input_count_array[node_id] != 0) { continue; }

            const auto &cur_node = storage_ptr->node(node_id);
            cur_node.for_each_output([&](BasicNode *next_op_node){
                if (next_op_node->graph_id != id) { return; }

                const NodeID node_id = next_op_node->id;
                const NodeID op_id   = node_id ^ GraphStorage::OP_NODE_ID_MASK;
                op_input_count_array[op_id]--;

                if (op_input_count_array[op_id] == 0) {
                    op_queue.push(static_cast<OperatorNode *>(next_op_node));
                }
            });
        }

        while (!op_queue.empty()) {
            OperatorNode *cur_node_ptr = op_queue.front();
            op_queue.pop();

            // Propagate the infection
            spy_info("propagate {}", cur_node_ptr->name);
            cur_node_ptr->propagate();

            // Get the next node and push into the queue
            cur_node_ptr->for_each_output([&](BasicNode *output_ptr){
                if (output_ptr->graph_id != id) { return; }

                output_ptr->for_each_output([&](BasicNode *next_op_node){
                    if (next_op_node->graph_id != id) { return; }

                    const NodeID node_id = next_op_node->id;
                    const NodeID op_id   = node_id ^ GraphStorage::OP_NODE_ID_MASK;
                    op_input_count_array[op_id]--;

                    if (op_input_count_array[op_id] == 0) {
                        op_queue.push(static_cast<OperatorNode *>(next_op_node));
                    }
                });
            });
        }
    }

    std::map<std::string_view, std::string> Graph::property() const {
        return {
            { "id",     std::to_string(id) }
        };
    }

} // namespace spy