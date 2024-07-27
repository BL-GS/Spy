#pragma once

#include "scheduler/scheduler.h"
#include "distributor/distributor.h"

/*
 * This kind of distributor does not divide graph and only the last backend added will execute the task.
 */

namespace spy {

   class SimpleGraphDistributor final: public AbstractGraphDistributor {
   private:
      std::unique_ptr<GraphScheduler> scheduler_ptr_;

      Graph *graph_ptr_ = nullptr;

   public:
      SimpleGraphDistributor() = default;

      ~SimpleGraphDistributor() noexcept override = default;

   public:
      bool add_backend(AbstractBackend *backend_ptr, const std::string_view sche_policy) override;

      void prepare_graph(Graph *graph_ptr) override;

      void execute() override;
   };
 
} // namespace spy
