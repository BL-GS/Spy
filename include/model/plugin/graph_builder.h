#pragma once

#include <magic_enum.hpp>

#include "model/config.h"
#include "model/file/config.h"
#include "operator/type.h"
#include "operator/operator.h"
#include "graph/graph.h"

namespace spy {

    /*!
     * @brief Utilities for building the graph
     */
    struct GraphBuilder {

        /*!
         * @brief Create a operation stream with inputs and create the output `Variable` tensor accordingly.
         */
        template<OperatorType T_op_type, class ...Args>
        static NodeCredit make_stream(Graph &graph, const std::string_view name, const std::initializer_list<NodeCredit> &inputs, Args &&...args) {
            const NodeCredit op_credit = graph.alloc_node<OperatorDefinition<T_op_type>>(
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

        template<OperatorType T_op_type, class ...Args>
        static NodeCredit make_stream_with_deps(Graph &graph, const std::string_view name, const std::initializer_list<NodeCredit> &inputs, const std::initializer_list<NodeCredit> &deps, Args &&...args) {
            const NodeCredit op_credit = graph.alloc_node<OperatorDefinition<T_op_type>>(
                    name,                        // Basic parameters
                    std::forward<Args>(args)...  // Additional operator parameters
            );
            // Build input
            for (NodeCredit input_credit: inputs) {
                graph.connect(input_credit, op_credit);
            }
            // Create output tensor
            const NodeCredit out_credit = create_variable_tensor<T_op_type>(graph, op_credit, std::string(name) + "- out");
            // TODO: hack
            for (NodeCredit dep_credit: deps) {
                graph.connect(dep_credit, op_credit);
            }
            // Build output
            graph.connect(op_credit, out_credit);
            return out_credit;
        }
        
        /*!
         * @brief Create a operation stream with inputs and output.
         */
        template<OperatorType T_op_type, class ...Args>
        static void make_determined_stream(Graph &graph, const std::string_view name, const std::initializer_list<NodeCredit> &inputs, const NodeCredit output, Args &&...args) {
            const NodeCredit op_credit = graph.alloc_node<OperatorDefinition<T_op_type>>(
                    name,                        // Basic parameters
                    std::forward<Args>(args)...  // Additional operator parameters
            );
            // Build input
            for (NodeCredit input_credit: inputs) {
                graph.connect(input_credit, op_credit);
            }
            // Build output
            graph.connect(op_credit, output);
        }

        /*!
         * @brief Create a tensor for input marked as `Constant`.
         * @note The scheduler SHOULD NOT allocate or deallocate it at runtime.
         * @note The user should set the data_ptr_ of tensor manually before executing.
         */
        static NodeCredit create_input_tensor(Graph &graph, const std::string_view tensor_name, 
                NumberType number_type, uint32_t dim, const Shape::DimensionArray &num_element, void *data_ptr)  {
            
            const Shape shape(dim, num_element, number_type);
            const NodeCredit node_credit = graph.alloc_node<DataNode>(tensor_name, DataNodeType::Constant, shape, data_ptr);

            graph.set_start(node_credit);
            return node_credit;
        }

        /*!
         * @brief Create a tensor for output marked as `Constant`.
         * @note The scheduler SHOULD NOT allocate or deallocate it at runtime.
         * @note The user should set the data_ptr_ of tensor manually before executing.
         */
        static NodeCredit create_output_tensor(Graph &graph, const std::string_view tensor_name, 
                NumberType number_type, uint32_t dim, const Shape::DimensionArray &num_element, void *data_ptr)  {
            
            const Shape shape(dim, num_element, number_type);
            const NodeCredit node_credit = graph.alloc_node<DataNode>(tensor_name, DataNodeType::Constant, shape, data_ptr);

			graph.set_end(node_credit);
            return node_credit;
        }


        /*!
         * @brief Create a tensor for constant tensor.
         * @note The scheduler SHOULD NOT allocate or deallocate it at runtime.
         * @note The user should set the data_ptr_ of tensor manually before executing.
         */
        static NodeCredit create_constant_tensor(Graph &graph, const std::string &tensor_name, 
                NumberType number_type, uint32_t dim, const Shape::DimensionArray &num_element, void *data_ptr, bool no_dequantize)  {
            
            const Shape shape(dim, num_element, number_type);
            const NodeCredit node_credit = graph.alloc_node<DataNode>(tensor_name, DataNodeType::Constant, shape, data_ptr);

			if (!no_dequantize) {
				if (number_type != NumberType::FP32) {
					spy_assert(number_type == NumberType::Q8_0);
					return make_stream<OperatorType::Quantize>(graph, tensor_name + ".fp32", { node_credit }, NumberType::FP32);
				}
			}

            graph.set_start(node_credit);
            return node_credit;
        }

        /*!
         * @brief Create a tensor for variable data. 
         * @note The scheduler will allocate it if needed and may deallocate it at runtime if it is not a view.
         */
        template<OperatorType T_op_type>
        static NodeCredit create_variable_tensor(Graph &graph, NodeCredit op_node_credit, const std::string_view tensor_name) {

            using OperatorDefNode = OperatorDefinition<T_op_type>;
            
            OperatorDefNode *   op_node_ptr  = graph.get_node_content<OperatorDefNode>(op_node_credit);
            const OperatorType  op_type      = op_node_ptr->op_type;
            spy_assert(op_type == T_op_type, 
                "Trying to get result tensor with different op_type (assign: {}, template: {})", 
                magic_enum::enum_name(op_type), magic_enum::enum_name(T_op_type));

            auto inputs = op_node_ptr->get_input();
            const DataNodeType data_type = is_view(op_type) ? DataNodeType::View : DataNodeType::Variable;
            const NodeCredit node_credit = graph.alloc_node<DataNode>(
                tensor_name, 
                data_type,
                op_node_ptr->deduce_result()
            );

            return node_credit;
        }

        
        /*!
         * @brief Create a tensor for constant data. The location and the size remain fixed at runtime.
         * @note The tensor will be set as input of the graph.
         * @note The scheduler SHOULD NOT allocate or deallocate it.
         */
        static NodeCredit create_constant_tensor(const GGUFContext &context, Graph &graph, 
                ModelTensorType tensor_type, uint32_t dim, const Shape::DimensionArray &num_element, 
                const std::string_view suffix, int layer_idx = -1, bool no_dequantize = false) {

            const TensorNameTable &tensor_table = context.model_tensor_table;
            const std::string      tensor_name  = (layer_idx == -1) ? tensor_table(tensor_type) : tensor_table(tensor_type, layer_idx);
            const std::string      concat_name  = tensor_name + suffix.data();

            // Get the metadata of the tensor
            const auto &info_map  = context.infos;
            const auto &info_iter = info_map.find(concat_name);
            if (info_iter == info_map.end()) { return Graph::INVALID_NODE_CREDIT; }
            const auto &info = info_iter->second;

            const NumberType number_type = info.type;
            const Shape shape(dim, num_element, number_type);

            const NodeCredit node_credit = graph.alloc_node<DataNode>(concat_name, DataNodeType::Constant, shape, nullptr);

            graph.set_start(node_credit);
	        // Dequantize weight tensor
	        // TODO: hacking and inefficient
	        spy_assert(info.data_ptr != nullptr, "The pointer to the weight is nullptr: {}", concat_name);
	        graph.get_node_content<DataNode>(node_credit)->tensor.set_data_ptr(const_cast<void *>(info.data_ptr));

			if (!no_dequantize) {
				if (number_type != NumberType::FP32) {
					spy_assert(number_type == NumberType::Q8_0);
					return make_stream<OperatorType::Quantize>(graph, concat_name + ".fp32", { node_credit }, NumberType::FP32);
				}
			}

            return node_credit;
        }

        /*!
         * @brief Create a tensor for input marked as `Buffered`.
         * @note The scheduler SHOULD allocate it at the first time and SHOULD NOT allocate or deallocate it at runtime.
         */
        static NodeCredit create_buffer_tensor(Graph &graph, const std::string_view tensor_name, 
                NumberType number_type, uint32_t dim, const Shape::DimensionArray &num_element, void *data_ptr) {

            const Shape shape(dim, num_element, number_type);
            const NodeCredit node_credit = graph.alloc_node<DataNode>(tensor_name, DataNodeType::Buffered, shape, data_ptr);

            graph.set_start(node_credit);
            return node_credit;
        }

        /*!
         * @brief Create a constant tensor for weight data of models.
         */
        static NodeCredit create_weight_tensor(const GGUFContext &context, Graph &graph, 
                ModelTensorType tensor_type, uint32_t dim, const Shape::DimensionArray &num_element, int layer_idx = -1, bool no_dequantize = false) {
            return create_constant_tensor(context, graph, tensor_type, dim, num_element, ".weight", layer_idx, no_dequantize);
        }

        /*!
         * @brief Create a constant tensor for bias data of models.
         */
        static NodeCredit create_bias_tensor(const GGUFContext &context, Graph &graph, 
                ModelTensorType tensor_type, uint32_t dim, const Shape::DimensionArray &num_element, int layer_idx = -1) {
            return create_constant_tensor(context, graph, tensor_type, dim, num_element, ".bias", layer_idx);
        }

    };

} // namespace spy