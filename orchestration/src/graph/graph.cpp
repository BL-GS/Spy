#include <string>
#include <string_view>
#include <queue>
#include <map>

#include "graph/graph.h"

namespace spy {

    std::map<std::string_view, std::string> Graph::property() const {
        return {
            { "id",     std::to_string(id) }
        };
    }

    void GraphStorage::propagate() const {
        auto data_input_count_array = get_data_input_count();
        auto op_input_count_array   = get_op_input_count();

        std::queue<OperatorNode *> op_queue;

        for (NodeID input_id: input_node_id_array_) {
            OperatorNode *entry_point = op_node_array_[id2idx(input_id)].get();

            entry_point->propagate();
            entry_point->for_each_output([&](BasicNode *input_node_ptr){
                const NodeID node_id = input_node_ptr->id;
                const int res = --data_input_count_array[node_id];

                spy_assert(res == 0, "invalid number of input tensors: {} (expect: 1)", res);
            });
        }

        for (size_t node_id = 0; node_id < data_input_count_array.size(); ++node_id) {
            if (data_input_count_array[node_id] != 0) { continue; }

            const auto &cur_node = node(node_id);
            cur_node.for_each_output([&](BasicNode *next_op_node){
                const NodeID node_id = next_op_node->id;
                const NodeID op_id   = id2idx(node_id);
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
            // spy_debug(DebugFlag::Graph, "propagate {}", cur_node_ptr->name);
            cur_node_ptr->propagate();

            // Get the next node and push into the queue
            cur_node_ptr->for_each_output([&](BasicNode *node_ptr){
                DataNode *output_ptr = dynamic_cast<DataNode *>(node_ptr);

                output_ptr->for_each_output([&](BasicNode *node_ptr){
                    OperatorNode *next_op_node = dynamic_cast<OperatorNode *>(node_ptr);

                    const NodeID node_id = next_op_node->id;
                    const NodeID op_id   = id2idx(node_id);
                    op_input_count_array[op_id]--;

                    if (op_input_count_array[op_id] == 0) {
                        op_queue.push(static_cast<OperatorNode *>(next_op_node));
                    }
                });
            });
        }
    }

} // namespace spy