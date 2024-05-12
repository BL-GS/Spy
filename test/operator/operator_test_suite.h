#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include <memory>
#include <gtest/gtest.h>

#include "model/config.h"
#include "number/tensor.h"
#include "operator/type.h"
#include "graph/graph.h"

namespace spy {

    struct OperatorTestGraph {
    public:
        OperatorTestGraph() = default;

        ~OperatorTestGraph() = default;

    public: /* Graph component generation */
        template<OperatorType T_op_type, class ...Args>
        NodeCredit make_stream(Graph &graph, const std::string_view name, const std::initializer_list<NodeCredit> &inputs, Args &&...args) const {
            const NodeCredit op_credit = graph.alloc_node<OperatorNodeImpl<T_op_type, BackendType::CPU>>(
                    name,                        // Basic parameters
                    std::forward<Args>(args)...  // Additional operator parameters
            );
            // Build input
            for (NodeCredit input_credit: inputs) {
                graph.connect(input_credit, op_credit);
            }
            // Create output tensor
            const NodeCredit out_credit = create_variable_tensor<T_op_type>(graph, op_credit, std::string(name) + "- out");
            // Build output
            graph.connect(op_credit, out_credit);
            return out_credit;
        }

        static NodeCredit create_input_tensor(Graph &graph, const std::string_view tensor_name, 
                NumberType number_type, uint32_t dim, const Shape::DimensionArray &num_element)  {
            
            const Shape shape(dim, num_element, number_type);
            const NodeCredit node_credit = graph.alloc_node<DataNode>(tensor_name, shape, nullptr);

            // The constant tensor should have been prepared as the system start.
            // So we set it as one of the start nodes.
            graph.set_start(node_credit);
            return node_credit;
        }

        template<OperatorType T_op_type>
        NodeCredit create_variable_tensor(Graph &graph, NodeCredit op_node_credit, const std::string_view tensor_name) const {
            
            const OperatorNode *op_node_ptr  = graph.get_node_content<OperatorNode>(op_node_credit);
            const OperatorType  op_type      = op_node_ptr->op_type;
            spy_assert(op_type == T_op_type, 
                "Trying to get result tensor with different op_type (assign: {}, template: {})", 
                magic_enum::enum_name(op_type), magic_enum::enum_name(T_op_type));

            auto inputs = op_node_ptr->get_input();
            const NodeCredit node_credit = graph.alloc_node<DataNode>(
                tensor_name, 
                op_node_ptr->get_result_tensor() const
            );

            return node_credit;
        }

    };

} // namespace spy