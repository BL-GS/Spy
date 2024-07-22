#include <iostream>
#include <fstream>
#include <string>

#include "util/shell/logger.h"
#include "adapter/adapter.h"
#include "graph/graph.h"
#include "graph/dot.h"
#include "llm/model/model.h"

using namespace spy;

int main(int argc, char **argv) {
    spy_assert(argc == 2);
    std::string filename = argv[1];

    auto context = FileAdapterFactory
        ::build_file_adapter("gguf")
        ->init_from_file(filename);

    std::cout << context.to_string() << std::endl;

    	// The hyper parameters defined by user will overwrite that read from model file.
	HyperParam hyper_param {
		.num_context = 32
	};

    GraphStorage storage;
    Graph graph(0, storage);
    auto model = ModelBuilder::build_model("llama", std::move(context), hyper_param);

	ModelIO model_io;
	model_io.add(0, 0, { 0 }, false);
	model->build_graph(graph, model_io);

	DotGenerator dot_generator;
    std::ofstream graph_stream("graph.dot", std::ios_base::ate);
    if (!graph_stream.good()) {
        std::cout << "failed open file: graph.dot\n";
    } 
	dot_generator.print_graph(graph_stream, graph);
    graph_stream.flush();
    graph_stream.close();
    
    return 0;
}