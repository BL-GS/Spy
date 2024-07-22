#pragma once

#include <vector>
#include <string>
#include <map>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "util/type/printable.h"

namespace spy {

	using NodeID  = uint32_t;
	using GraphID = uint32_t;

	static constexpr NodeID  INVALID_NODE_ID  = std::numeric_limits<NodeID>::max();
	static constexpr GraphID INVALID_GRAPH_ID = std::numeric_limits<NodeID>::max();

    struct BasicNode: PropertyInterface {
	public:
		/// The unique id of the node
		NodeID 				id			= INVALID_NODE_ID;
		/// The unique id to the underlying graph
		GraphID				graph_id 	= INVALID_GRAPH_ID;
		/// The name of the node
		std::string 		name		= "Unknown";

		bool				active  	= true;

		std::vector<BasicNode *> input_list;

		std::vector<BasicNode *> output_list;

	public:
		BasicNode() 		 = default;

		virtual ~BasicNode() = default;

	public:
		/*!
		 * @brief Add nodes as the input of this node
		 * @return The index of the last one
		 */
		template<class ...Args>
		size_t add_input(BasicNode *node_ptr, Args &&...args) {
			// Add node_ptr
			const size_t idx = input_list.size();
			input_list.push_back(node_ptr);
			node_ptr->output_list.push_back(this);
			// Add left input nodes
			if constexpr (sizeof ...(args) != 0)  { 
				return add_input(std::forward<Args>(args)...); 
			}
			return idx; 
		}
  
		/*!
		 * @brief Add nodes as the output of this node
		 * @return The index of the last one
		 */
		template<class ...Args>
		size_t add_output(BasicNode *node_ptr, Args &&...args) {
			// Add node_ptr
			const size_t idx = output_list.size();
			output_list.push_back(node_ptr);
			node_ptr->input_list.push_back(this);
			// Add left output nodes
			if constexpr (sizeof ...(args) != 0)  { 
				return add_input(std::forward<Args>(args)...); 
			}
			return idx; 
		}

	public:
		size_t num_input()  const { 
			return output_list.size(); 
		}

		size_t num_output() const { 
			return output_list.size(); 
		}

		template<class T_Node = BasicNode>
		T_Node *input(size_t idx) const { 
			spy_assert_debug(idx < num_input(), 
				"Invalid index of inputs: {} (max: {})", idx, num_input() - 1);
			return static_cast<T_Node *>(input_list[idx]);
		}

		template<class T_Node = BasicNode>
		T_Node *output(size_t idx) const { 
			spy_assert_debug(idx < num_input(), 
				"Invalid index of outputs: {} (max: {})", idx, num_input() - 1);
			return static_cast<T_Node *>(output_list[idx]); 
		}

		template<class T_Node = BasicNode, class T_Func>
		void for_each_input(T_Func &&func) const {
			for (size_t i = 0; i < num_input(); ++i) {
				func(input<T_Node>(i));
			}
		}

		template<class T_Node = BasicNode, class T_Func>
		void for_each_output(T_Func &&func) const {
			for (size_t i = 0; i < num_output(); ++i) {
				func(output<T_Node>(i));
			}
		}

	public:
		std::map<std::string_view, std::string> property() const override {
			std::map<std::string_view, std::string> property = {
				{ "id", 		std::to_string(id) 		},
				{ "name", 		name 							},
				{ "active",		active ? "true" : "false"		}
			};
			return property;
		}
	};

} // namespace spy