#pragma once

#ifndef BACKEND_DISTRIBUTOR_HEADER_MACRO
	#warning "Do not include simple_distributor.h manually, please use distributor/distributor.h instead."
#endif // BACKEND_DISTRIBUTOR_HEADER_MACRO

#include <memory>

#include "graph/graph.h"
#include "scheduler/scheduler.h"
#include "distributor/distributor.h"

/*
 * This kind of distributor does not divide graph and only the last backend added will execute the task.
 */

namespace spy {

   class SimpleGraphDistributor final: public GraphDistributor {
   private:
      std::unique_ptr<GraphScheduler> scheduler_ptr_;

      Graph *graph_ptr_ = nullptr;

   public:
      SimpleGraphDistributor(ModelLoader *loader_ptr): GraphDistributor(loader_ptr) {}

      ~SimpleGraphDistributor() noexcept override = default;

   public:
      bool add_backend(Backend *backend_ptr, const std::string_view sche_policy) override {
         if (scheduler_ptr_ != nullptr) {
               spy_warn("trying to overwrite the backend, only the last one will be used");
         }

         scheduler_ptr_ = GraphSchedulerBuilder::build_scheduler(sche_policy, backend_ptr);
         return true;         
      }

      void prepare_graph(Graph *graph_ptr) override {
         if (graph_ptr_ != nullptr) {
               spy_warn("trying to overwrite the graph");
         }
         graph_ptr_ = graph_ptr;         
      }

      void execute() override {
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
   };
 
} // namespace spy
