#pragma once

#include <string>
#include <set>
#include <queue>
#include <sstream>

#include "graph/data_node.h"
#include "graph/op_node.h"
#include "graph/graph.h"

namespace spy {

    struct DotGenerator {
    public:
        template<class T_Stream>
        static void print_graph(T_Stream &output, const Graph &graph) {
            const GraphID graph_id = graph.id;

            std::set<NodeID> visit_set;

            output << "digraph graph_"  << std::to_string(graph_id) << "{\n";

            // Print all nodes in the graph
            std::queue<const BasicNode *> node_queue;
            for (const BasicNode *entry_point: graph.entry_point_array) {
                node_queue.push(entry_point);
            }
            while (!node_queue.empty()) {
                const BasicNode *cur_node_ptr = node_queue.front();
                node_queue.pop();

                if (visit_set.find(cur_node_ptr->id) != visit_set.end()) { continue; }
                visit_set.insert(cur_node_ptr->id);

                output << print_node(cur_node_ptr)
                       << print_connection(cur_node_ptr);

                for (size_t i = 0; i < cur_node_ptr->num_output(); ++i) {
                    const BasicNode *output_node_ptr = cur_node_ptr->output(i);
                    const NodeID to_graph_id = output_node_ptr->graph_id;
                    if (to_graph_id == graph_id) {
                        node_queue.push(output_node_ptr);
                    }
                }
            }

            output << "}\n";
        }

        static std::string print_graph(const Graph &graph) {
            std::stringstream output;
            print_graph(output, graph);
            return output.str();
        }

    protected:
        static std::string print_node(const BasicNode *node_ptr) {
            const NodeID id = node_ptr->id;
            const auto prop = node_ptr->property();

            std::string res = std::to_string(id);

            {
                res += " [shape=record, label=\"{<head>";

                for (const auto &[key, value]: prop) {
                    res += key; 
                    res += ": ";
                    res += value;
                    res += '|';
                }
                res.pop_back();
                res += "}\"";

                if (typeid(*node_ptr) == typeid(DataNode)) {
                    res += "color=cadetblue";
                } else {
                    res += "color=bisque4";
                }
            }

            return res + " ];\n";
        }

        static std::string print_connection(const BasicNode *node_ptr) {
            const NodeID id          = node_ptr->id;
            const std::string id_str = std::to_string(id);
            std::string res;

            for (size_t i = 0; i < node_ptr->num_output(); ++i) {
                const BasicNode *output_node_ptr = node_ptr->output(i);
                const NodeID to_id = output_node_ptr->id;
                res += id_str + " -> " + std::to_string(to_id) + ";\n";
            }
            return res + '\n';
        }

    };

} // namespace spy