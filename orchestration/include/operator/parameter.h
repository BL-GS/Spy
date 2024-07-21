#pragma once

#include <optional>

#include "util/shell/logger.h"

namespace spy {

    template<class T>
    class OperatorParameter {
    public:
        using Value      = T;
        using RefPointer = const T*;

    private:
        /// The content
        std::optional<Value>  val_;
        /// The reference of value
        RefPointer ref_ptr_ = nullptr;

    public:
        OperatorParameter() = default;

        OperatorParameter(const T &val): val_(val) {}

        OperatorParameter(RefPointer ref_ptr): ref_ptr_(ref_ptr) {}

    public:
        /*!
         * @brief Track the reference and update the value
         * @return The reference to the updated value
         */
        const Value &track_ref() {
            spy_assert(is_ref());
            RefPointer ref_ptr = get_ref();
            spy_assert_debug(ref_ptr != nullptr, "trying to track an empty reference");
            val_ = *ref_ptr;
            return val_.value();
        }

        const Value &track_ref_if_needed() {
            if (is_ref()) { return track_ref(); }
            spy_assert(is_val());
            return get_val();
        }

    public:
        /*!
         * @brief Bind value with a remote parameter.
         * @param ref The reference to the reference to bind on
         */
        void set_ref(RefPointer ref_ptr) {
            ref_ptr_ = ref_ptr;
        }

        /*!
         * @brief Set value with certain value
         * @param value The certain value to copy
         */
        void set_val(const Value &value) {
            ref_ptr_ = nullptr;
            val_ = value;
        }

        RefPointer get_ref() const {
          spy_assert_debug(is_ref(), "trying to get reference from assigned parameter");
          return ref_ptr_;
        }

        const Value &get_val() const {
          spy_assert_debug(is_val(), "trying to get value from parameter non-assigned");
          return val_.value();
        }

      public:
        bool is_ref() const { return ref_ptr_ != nullptr; }

        bool is_val() const { return val_.has_value(); }

    };

} // namespace spy