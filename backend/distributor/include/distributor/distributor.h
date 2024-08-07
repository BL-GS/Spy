#pragma once

#include <memory>
#include <string_view>

namespace spy {

    class Backend;
    class Graph;
    class ModelLoader;

    class GraphDistributor {
    protected:
        ModelLoader *loader_ptr_ = nullptr;

    public:
        GraphDistributor(ModelLoader *loader_ptr): loader_ptr_(loader_ptr) {}

        virtual ~GraphDistributor() noexcept = default;

    public:
        /*!
         * @brief Add a new backend for workload offloading.
         * @param[in] backend_ptr The pointer to the backend.
         * @return true on success; otherwise failed.
         */
        virtual bool add_backend(Backend *backend_ptr, const std::string_view sche_policy) = 0;

        /*!
         * @brief Prepare graph for every backend
         * @param[in] graph_ptr The pointer to the original graph
         * @note This process cannot be called for multiple times.
         */
        virtual void prepare_graph(Graph *graph_ptr) = 0;

        /*!
         * @brief Distribute divided graphs into different backends and execute them.
         */
        virtual void execute() = 0;
    };

    class GraphDistributorFactory {
    public:
        /*!
         * @brief Build up a distributor for graph divide and task distribution.
         * @param[in] policy The name of policy (greedy).
         * @note Please infer to the source code of policy implementation for the detail of effect.
         */
        static std::unique_ptr<GraphDistributor> build_graph_distributor(std::string_view policy, ModelLoader *loader_ptr);

    };

} // namespace spy