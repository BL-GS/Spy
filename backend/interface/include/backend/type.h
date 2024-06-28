#pragma once

namespace spy {

    enum class BackendType {
        /// The data is accessible by host who control all of the devices.
        /// Data in this backend, can be transferred or used directly to the device.
        /// e.g. CPU memory
        Host,
        /// The data is of the same accessibility as the Host.
        /// e.g. NPU memory
        Share,
        /// The data is only accessible by the device itself.
        /// Data in this backend cannnot be transferred to other devices
        /// e.g. GPU memory
        Device,

        Unknown
    };

} // namespace spy