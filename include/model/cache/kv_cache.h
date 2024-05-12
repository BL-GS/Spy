#pragma once

#include <cstdint>
#include <vector>

#include "util/shell/logger.h"
#include "number/tensor.h"

namespace spy {

    struct KVCache {
    public:
        size_t head = 0;

        std::vector<Tensor> k_cache;
        std::vector<Tensor> v_cache;

        std::vector<std::unique_ptr<uint8_t []>> pointer_storage;

    public:
        KVCache() = default;

        ~KVCache() noexcept {
            spy_info("Delete kv cache");
        }

    public:
        void reserve(uint32_t n_embd_k_gqa, uint32_t n_embd_v_gqa, uint32_t kv_size, uint32_t num_layer) {
            k_cache.reserve(num_layer);
            v_cache.reserve(num_layer);
            pointer_storage.reserve(2 * num_layer);

            for (uint32_t i = k_cache.size(); i < num_layer; ++i) {
                const size_t k_num  = n_embd_k_gqa * kv_size;
                const Shape k_shape{{k_num}, NumberType::FP16};
                const size_t k_size = k_shape.total_size();
                uint8_t *k_data = new uint8_t[k_size];
                k_cache.emplace_back(k_shape, k_data);
                pointer_storage.emplace_back(k_data);
            }

            for (uint32_t i = v_cache.size(); i < num_layer; ++i) {
                const size_t v_num  = n_embd_v_gqa * kv_size;
                const Shape v_shape{{v_num}, NumberType::FP16};
                const size_t v_size = v_shape.total_size();
                uint8_t *v_data = new uint8_t[v_size];
                v_cache.emplace_back(v_shape, v_data);
                pointer_storage.emplace_back(v_data);
            }
        }

        void step(size_t n) { head += n; }
    };

} // namespace spy