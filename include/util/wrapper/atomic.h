#pragma once

#include <atomic>
#include <memory>
#include <any>
#include <condition_variable>

namespace spy {

    /*!
     * @brief Wrapper for atomic unit in container.
     * In std container, underlying type should have copy constructor.
     * And std::atomic do not speicify the implementation.
     * This class aims to simply wrap std::atomic for convenience.
     */
    template <typename T>
    struct AtomWrapper {
    public:
        std::atomic<T> value;

    public:
        AtomWrapper() = default;

        AtomWrapper(const AtomWrapper &other) :value(other.value.load()) {}

        AtomWrapper(const std::atomic<T> &other) :value(other.load()) {}

        AtomWrapper(const T &other) :value(other) {}

        AtomWrapper &operator=(const AtomWrapper &other) {
            value.store(other.value.load());
        }

        AtomWrapper &operator=(const std::atomic<T> &other) {
            value.store(other.load());
        }

        AtomWrapper &operator=(const T &other) {
            value.store(other);
        }

        T load(std::memory_order order = std::memory_order_acq_rel) { return value.load(order); }

        void store(const T new_val, std::memory_order order = std::memory_order_acq_rel) { value.store(new_val, order); }

        T operator++() { return ++value; }

        T operator++(int) { return value++; }

        T operator--() { return --value; }

        T operator--(int) { return value--; }
    };

    /*!
     * @brief Wrapper for atomic unit in container.
     * In std container, underlying type should have copy constructor.
     * And std::atomic do not speicify the implementation.
     * This class aims to simply wrap std::atomic for convenience.
     */
    template <typename T>
    struct RelaxedAtomWrapper {
    public:
        std::atomic<T> value;

    public:
        RelaxedAtomWrapper() = default;

        RelaxedAtomWrapper(const RelaxedAtomWrapper &other) :value(other.value.load(std::memory_order_relaxed)) {}
   
        RelaxedAtomWrapper(const std::atomic<T> &other) :value(other.load(std::memory_order_relaxed)) {}

        RelaxedAtomWrapper(const T &other) :value(other) {}

        RelaxedAtomWrapper &operator=(const RelaxedAtomWrapper &other) {
            value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }

        RelaxedAtomWrapper &operator=(const std::atomic<T> &other) {
            value.store(other.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }

        RelaxedAtomWrapper &operator=(const T &other) {
            value.store(other, std::memory_order_relaxed);
        }

        T load(std::memory_order order = std::memory_order_acq_rel) { return value.load(order); }

        void store(const T new_val, std::memory_order order = std::memory_order_acq_rel) { value.store(new_val, order); }

        T operator++() { return ++value; }

        T operator++(int) { return value++; }

        T operator--() { return --value; }

        T operator--(int) { return value--; }
    };

} // namespace spy