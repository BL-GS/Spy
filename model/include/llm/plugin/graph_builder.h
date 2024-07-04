#pragma once

#include <magic_enum.hpp>

#include "operator/type.h"
#include "operator/operator.h"
#include "graph/graph.h"
#include "metadata.h"
#include "llm/config.h"

namespace spy {

    /*!
     * @brief Utilities for building the graph
     */
    struct GraphBuilder {

        /*!
         * @brief Create a operation stream with inputs and create the output `Variable` tensor accordingly.
         */
        template<OperatorType T_op_type, class ...Args>
        static DataNode *make_stream(Graph &graph, TensorType output_tensor_type, int layer_id, int expert_id,
									 const std::initializer_list<DataNode *> &inputs,
									 Args &&...args) {
            auto *op_node_ptr = graph.alloc_node<OperatorDefinition<T_op_type>>(std::forward<Args>(args)...);
            // Build input
            for (DataNode *input_node_ptr: inputs) { graph.connect(input_node_ptr, op_node_ptr); }
            // Create output tensor
            DataNode *output_node_ptr = create_variable_tensor<T_op_type>(graph, op_node_ptr,
                output_tensor_type, layer_id, expert_id
            );
            // Build output
            graph.connect(op_node_ptr, output_node_ptr);
            return output_node_ptr;
        }

        /*!
         * @brief Create a operation stream with inputs and output.
         */
        template<OperatorType T_op_type, class ...Args>
        static void make_determined_stream(Graph &graph,
										   const std::initializer_list<DataNode *> &inputs,
										   DataNode *output_node_ptr, Args &&...args) {
            auto *op_node_ptr = graph.alloc_node<OperatorDefinition<T_op_type>>(std::forward<Args>(args)...);
            // Build input
            for (DataNode *input_node_ptr: inputs) {
                graph.connect(input_node_ptr, op_node_ptr);
            }
            // Build output
            graph.connect(op_node_ptr, output_node_ptr);
        }

        /*!
         * @brief Create a tensor for input marked as `Constant`.
         * @note The scheduler SHOULD NOT allocate or deallocate it at runtime.
         * @note The user should set the data_ptr_ of tensor manually before executing.
         */
        static DataNode *create_input_tensor(Graph &graph,
                NumberType number_type, uint32_t dim, const Shape::DimensionArray &num_element, void *data_ptr,
                TensorType tensor_type, int layer_id = -1, int expert_id = -1) {
            
            const Shape shape(dim, num_element, number_type);
            const DataNodeProperty property {
                .node_type   = DataNodeType::Constant,
                .tensor_type = tensor_type,
                .weight_type = WeightType::Input,
                .layer_id    = layer_id,
                .expert_id   = expert_id
            };

            DataNode *node_ptr = graph.alloc_node<DataNode>(property, shape, data_ptr);
            return node_ptr;
        }

        /*!
         * @brief Create a tensor for output marked as `Constant`.
         * @note The scheduler SHOULD NOT allocate or deallocate it at runtime.
         * @note The user should set the data_ptr_ of tensor manually before executing.
         */
        static DataNode *create_output_tensor(Graph &graph,
                NumberType number_type, uint32_t dim, const Shape::DimensionArray &num_element, void *data_ptr,
                TensorType tensor_type, int layer_id = -1, int expert_id = -1)  {
            
            const Shape shape(dim, num_element, number_type);
            const DataNodeProperty property {
                .node_type   = DataNodeType::Constant,
                .tensor_type = tensor_type,
                .weight_type = WeightType::None,
                .layer_id    = layer_id,
                .expert_id   = expert_id
            };
	        DataNode *node_ptr = graph.alloc_node<DataNode>(property, shape, data_ptr);
			return node_ptr;
        }

        /*!
         * @brief Create a tensor for variable data. 
         * @note The scheduler will allocate it if needed and may deallocate it at runtime if it is not a view.
         */
        template<OperatorType T_op_type>
        static DataNode *create_variable_tensor(Graph &graph, OperatorDefinition<T_op_type> *op_node_ptr,
            TensorType tensor_type, int layer_id = -1, int expert_id = -1) {

            const OperatorType  op_type      = op_node_ptr->op_type;
            spy_assert(op_type == T_op_type, 
                "Trying to get result tensor with different op_type (assign: {}, template: {})",
                op_type, T_op_type);

            const DataNodeProperty property {
                .node_type      = is_view(op_type) ? DataNodeType::View : DataNodeType::Variable,
                .tensor_type    = tensor_type,
                .weight_type    = is_view(op_type) ? WeightType::View : WeightType::Output,
                .layer_id       = layer_id,
                .expert_id      = expert_id
            };

            DataNode *node_ptr = graph.alloc_node<DataNode>(
                property,
                op_node_ptr->deduce_result()
            );

            return node_ptr;
        }

	    /*!
		 * @brief Create a tensor for constant data. The location and the size remain fixed at runtime.
		 * @note The tensor will be set as input of the graph.
		 * @note The scheduler SHOULD NOT allocate or deallocate it.
		 */
        static DataNode *create_constant_tensor(const ModelMetaContext &context, Graph &graph,
                uint32_t dim, const Shape::DimensionArray &num_element, 
                TensorType tensor_type, WeightType weight_type, int layer_id = -1, int expert_id = -1) {

            const DataNodeProperty property {
                .node_type   = DataNodeType::Constant,
                .tensor_type = tensor_type,
                .weight_type = weight_type,
                .layer_id    = layer_id,
                .expert_id   = expert_id
            };
            const std::string tensor_name = property.to_string();

            // Get the metadata of the tensor
            const auto &info_map  = context.infos;
            const auto &info_iter = info_map.find(tensor_name);
            if (info_iter == info_map.end()) { return nullptr; }
            const auto &info = info_iter->second;

            const NumberType number_type = info.type;
            const Shape shape(dim, num_element, number_type);

            DataNode *node_ptr = graph.alloc_node<DataNode>(property, shape, nullptr);

	        // Dequantize weight tensor
	        // TODO: hacking and inefficient
	        spy_assert(info.data_ptr != nullptr, "The pointer to the weight is nullptr: {}", tensor_name);
	        node_ptr->tensor.set_data_ptr(const_cast<void *>(info.data_ptr));

            return node_ptr;
        }

        /*!
         * @brief Create a tensor for input marked as `Buffered`.
         * @note The scheduler SHOULD allocate it at the first time and SHOULD NOT allocate or deallocate it at runtime.
         */
        static DataNode *create_buffer_tensor(Graph &graph,
                NumberType number_type, uint32_t dim, const Shape::DimensionArray &num_element, void *data_ptr,
                TensorType tensor_type, int layer_id = -1, int expert_id = -1) {

            const Shape shape(dim, num_element, number_type);
            const DataNodeProperty property {
                .node_type      = DataNodeType::Buffered,
                .tensor_type    = tensor_type,
                .weight_type    = WeightType::None,
                .layer_id       = layer_id,
                .expert_id      = expert_id
            };

            DataNode *node_ptr = graph.alloc_node<DataNode>(property, shape, data_ptr);
            return node_ptr;
        }

        /*!
         * @brief Create a constant tensor for weight data of models.
         */
        static DataNode *create_weight_tensor(const ModelMetaContext &context, Graph &graph,
                uint32_t dim, const Shape::DimensionArray &num_element, 
                TensorType tensor_type, int layer_idx = -1, int expert_id = -1) {
            return create_constant_tensor(context, graph, 
                dim, num_element,
                tensor_type, WeightType::Weight, layer_idx, expert_id
            );
        }

        /*!
         * @brief Create a constant tensor for bias data of models.
         */
        static DataNode *create_bias_tensor(const ModelMetaContext &context, Graph &graph,
                uint32_t dim, const Shape::DimensionArray &num_element, 
                TensorType tensor_type, int layer_idx = -1, int expert_id = -1) {
            return create_constant_tensor(context, graph, 
                dim, num_element,
                tensor_type, WeightType::Bias, layer_idx, expert_id
            );
        }

    };

} // namespace spy