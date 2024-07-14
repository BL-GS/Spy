#include <iostream>
#include <string>

#include "util/shell/logger.h"
#include "adapter/adapter.h"

using namespace spy;

int main(int argc, char **argv) {
    spy_assert(argc == 2);
    std::string filename = argv[1];

    auto file_adapter = FileAdapterFactory::build_file_adapter("gguf");
    file_adapter->init_from_file(filename);

    std::cout << file_adapter->context.to_string() << std::endl;
    
    return 0;
}