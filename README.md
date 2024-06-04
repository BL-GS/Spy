# Spy

This is a modular LLM Inference framework referring to [llama.cpp](https://github.com/ggerganov/llama.cpp)

> Although `llama.cpp` is very popular, the file organization and the function design make it quite hard to extend.
> Therefore, I'd like to reconstruct it (not at all) use modern C++ to make a easy-to-use ML/DL inference framework.

# Build

This project is only developed on Windows 11 and WSL2 (Ubuntu 22.04) with MSBuild 17.9.8 and CMake. 
It is experimented to build by `msys-clang64 18.1.4`, `MSVC 14.39` and `GCC 13.1.0`.
It depends on the following open-source libraries:
- [magic-enum](https://github.com/Neargye/magic_enum)
- [fmt](https://github.com/fmtlib/fmt)
- [spdlog](https://github.com/gabime/spdlog)
- [argparse](https://github.com/p-ranav/argparse)
- [concurrentqueue](https://github.com/cameron314/concurrentqueue)

## CMake + Vcpkg

It is quite easy to build with **CMake + vcpkg**. At first, install vcpkg according to its [documentation](https://github.com/microsoft/vcpkg).
1. Install the above-mentioned libraries:
    ```shell
    vcpkg install magic-enum fmt spdlog argparse concurrentqueue
    ```
2. Configure CMake with vcpkg (the toolchain file locates in `<installation directory of vcpkg>/scripts/buildsystems/vcpkg.cmake`):
    ```shell
    cmake -B build -DCMAKE_TOOLCHAIN_FILE=<installation directory of vcpkg>/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
    ```
3. Build the project
    ```shell
    cmake --build build --config Release -j
    ```
4. Launch the program
    ```shell
    ./build/spy -m <path-to-model> -n 32 -t 8 -c 32 -p "Once upon a time"
    ```
   
## XMake + Xrepo

It is easier to build with **XMake + Xrepo**. At first, install xmake according to its [documentation](https://xmake.io/#/getting_started).
1. Install the above-mentioned libraries:
   ```shell
   xrepo install magic-enum fmt spdlog argparse concurrentqueue
   ```
2. Configure XMake
   ```shell
   xmake f --mode=release -c
   ```
3. Build the project
   ```shell
   xmake --build
   ```
4. Launch the program (The path to the executable may be different, depending on the configuration and platform)
   ```shell
   ./build/linux/x86_64/release/spy -m <path-to-model> -n 32 -t 8 -c 32 -p "Once upon a time" 
   ```

# Model

Currently, it only support gguf model of LLaMa2-7b, which can be gained according to the documentation of [llama.cpp](https://github.com/ggerganov/llama.cpp).

# Backend

Currently, it only support CPU as the backend.

