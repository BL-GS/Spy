#pragma once

#include <memory>
#include <vector>
#include <forward_list>
#include <any>
#include <functional>
#include <magic_enum.hpp>

#include "operator/type.h"
#include "operator/operator.h"
#include "adapter/type.h"
#include "graph/graph.h"

namespace spy {

    /*!
     * @brief Utilities for building the graph
     */
    struct GraphBuilder {
    public:
        /// Listeners for updating params, which should be called after the member variables have been updated
        std::vector<std::function<void()>> listener_list;
        /// Parameter storage
        std::forward_list<std::any> param_list;

    public:
        template<class T, class ...Args>
        T &alloc_param(Args &&...args) {
            return std::any_cast<T &>(param_list.emplace_front(T{std::forward<Args>(args)...}));
        }

        template<class T_Func>
        void add_listener(T_Func &&func) {
            listener_list.emplace_back(func);
        }

        template<class T, class T_Func, class ...Args>
        T &add_param_listener(T_Func &&func, Args &&...args) {
            T &param = alloc_param<T>(std::forward<Args>(args)...);
            listener_list.emplace_back([&param, func](){ param = func(); });
            return param;
        }

        void notify_listeners() const {
            for (auto &listener: listener_list) { listener(); }
        }

    public:
        /*!
         * @brief Create a operation stream with inputs and create the output `Variable` tensor accordingly.
         */
        template<OperatorType T_op_type, class ...Args>
        static auto make_stream(Graph &graph, const std::string_view name, const DataNodeProperty &prop, Args &&...inputs) {
            auto &op_node = graph.alloc_node<OperatorDefinition<T_op_type>>();
            op_node.name = name;
            return op_node.deduce(graph, prop, std::forward<Args>(inputs)...);
        }

        /*!
         * @brief Create a operation stream with inputs and create the output `Variable` tensor accordingly.
         */
        template<OperatorType T_op_type, class ...Args>
        static auto make_augmented_stream(Graph &graph, const std::string_view name, const DataNodeProperty &prop,
									const OperatorDefinition<T_op_type>::Param &param, Args &&...inputs) {
            auto &op_node = graph.alloc_node<OperatorDefinition<T_op_type>>(param);
            op_node.name = name;
            return op_node.deduce(graph, prop, std::forward<Args>(inputs)...);
        }

        /*!
         * @brief Create a operation stream with inputs and create the output `Variable` tensor accordingly.
         * @param func A function for creating a latest parameter.
         */
        template<OperatorType T_op_type, class T_Func, class ...Args>
        auto make_dynamic_stream(Graph &graph, const std::string_view name, const DataNodeProperty &prop,
									T_Func &&func, Args &&...inputs) {
            using Param = OperatorDefinition<T_op_type>::Param;

            Param &param = add_param_listener<Param>(std::forward<T_Func>(func));
            // pass the pointer of parameter as consequent reference
            auto &op_node = graph.alloc_node<OperatorDefinition<T_op_type>>(std::addressof(param));
            op_node.name = name;
            return op_node.deduce(graph, prop, std::forward<Args>(inputs)...);
        }

        /*!
         * @brief Create a tensor 
         * @note The scheduler SHOULD NOT allocate or deallocate it at runtime.
         * @note The user should set the data_ptr_ of tensor manually before executing.
         */
        static DataNode *create_tensor(Graph &graph, const std::string_view name,
                const Shape &shape, const DataNodeProperty &prop, void *data_ptr) {
            
            DataNode &data_node = graph.alloc_node<DataNode>(prop, shape, data_ptr);
            data_node.name = name;
            return std::addressof(data_node);
        }

	    /*!
		 * @brief Create a tensor for constant data. The location and the size remain fixed at runtime.
		 * @note The tensor will be set as input of the graph.
		 * @note The scheduler SHOULD NOT allocate or deallocate it.
		 */
        static DataNode *create_constant_tensor(const ModelMetaContext &context, Graph &graph,
                const std::string_view tensor_base_name,
                const DataNodeProperty &prop, const std::string_view tensor_suffix = "weight") {

            const std::string tensor_name = make_tensor_name(tensor_base_name, prop, tensor_suffix);

            // Get the metadata of the tensor
            const auto &info_map  = context.infos;
            const auto &info_iter = info_map.find(tensor_name);
            if (info_iter == info_map.end()) { return nullptr; }
            const auto &info = info_iter->second;

            const NumberType number_type = info.type;
            const Shape shape(info.num_dim, info.num_element, number_type);

            DataNode &data_node = graph.alloc_node<DataNode>(prop, shape, info.data_ptr);
            return std::addressof(data_node);
        }

        static std::string make_tensor_name(const std::string_view tensor_base_name, const DataNodeProperty &prop, 
                const std::string_view tensor_suffix = "weight") {

            std::string tensor_name;
            if (prop.layer_id != -1 || prop.expert_id != -1) {
                tensor_name += "blk.";
            }
            if (prop.layer_id != -1) {
                tensor_name += std::to_string(prop.layer_id) + '.';
            }
            if (prop.expert_id != -1) {
                tensor_name += std::to_string(prop.expert_id) + '.';
            }

            return tensor_name
                    .append(tensor_base_name)
                    .append(".").append(tensor_suffix);
        }

    };

} // namespace spy