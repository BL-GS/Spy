#include <cstdint>
#include <random>
#include <vector>
#include <gtest/gtest.h>

#include "operator/type.h"
#include "operator_test_suite.h"
#include "backend/cpu/cpu.h"
#include "graph/scheduler.h"

using namespace spy;

template<OperatorType T_op_type>
void mat_op_test(const Shape &shape_0, const Shape &shape_1) {
	const size_t num_0 = shape_0.total_element();
	const size_t num_1 = shape_1.total_element();
	const size_t num_res = num_0 * shape_1.elements[1] / shape_0.elements[0];

	std::vector<float> operand_0(num_0, 0);
	std::vector<float> operand_1(num_1, 0);
	std::vector<float> result;
	std::vector<float> reference(num_res, 0);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist;

	for (size_t i = 0; i < num_0; ++i) { operand_0[i] = dist(gen); }
	for (size_t i = 0; i < num_1; ++i) { operand_1[i] = dist(gen); }

	const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
	const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

	for (size_t i01 = 0; i01 < ne01; ++i01) { // Select a row of operand 0
		for (size_t i11 = 0; i11 < ne11; ++i11) { // Select a col of operand 1
			float &res = reference[i11 * ne01 + i01];
			res = 0;

			for (size_t i00 = 0; i00 < ne00; ++i00) {
				const float src0 = operand_0[i01 * ne00 + i00];
				const float src1 = operand_1[i11 * ne00 + i00];
				res += src0 * src1;
			}
		}
	}

	DefaultCPUBackend cpu_backend(1, 0);
	Graph graph;
	OperatorTestGraph base;

	const NodeCredit input_operand_0 = base.create_input_tensor(graph, "Operand 0", NumberType::FP32, shape_0.dim, shape_0.elements);
	graph.get_node_content<DataNode>(input_operand_0)->tensor.set_data_ptr(operand_0.data());
	const NodeCredit input_operand_1 = base.create_input_tensor(graph, "Operand 1", NumberType::FP32, shape_1.dim, shape_1.elements);
	graph.get_node_content<DataNode>(input_operand_1)->tensor.set_data_ptr(operand_1.data());
	const NodeCredit output = base.make_stream<T_op_type>(graph, "Op", { input_operand_0, input_operand_1 });
	Tensor &output_tensor = graph.get_node_content<DataNode>(output)->tensor;
	result.resize(output_tensor.total_element(), 0);
	output_tensor.set_data_ptr(result.data());

	graph.set_end(output);

	GraphScheduler scheduler(&graph, &cpu_backend);
	scheduler.execute();

	for (int i = 0; i < num_res; ++i) {
		EXPECT_FLOAT_EQ(reference[i], result[i]);
	}
}

class BasicTest: public testing::Test {
public:
	static constexpr size_t MIN_DIM         = 1;
	static constexpr size_t MAX_DIM         = 2;
	static constexpr size_t MIN_ELEMENT     = 16;
	static constexpr size_t MAX_ELEMENT     = 1024;
	static constexpr size_t NUM_TEST        = 16;

	static_assert(MAX_DIM <= 2, "Remove after changing mat_op_test");

public:
	std::vector<std::pair<Shape, Shape>>                      test_shape_array;

public:

protected:
	void SetUp() override {
		std::random_device                      random_device;
		std::mt19937                            random_gen(random_device());

		std::uniform_int_distribution<size_t>   dim_rander(MIN_DIM, MAX_DIM);
		std::uniform_int_distribution<size_t>   ne_rander(MIN_ELEMENT, MAX_ELEMENT);

		for (size_t i = 0; i < NUM_TEST; ++i) {
			Shape::DimensionArray array;
			const size_t dim = dim_rander(random_gen);
			for (size_t d = 0; d < dim; ++d) {
				const size_t ne = ne_rander(random_gen);
				array[d] = ne;
			}
			Shape::DimensionArray array1 = array;
			for (size_t d = 1; d < dim; ++d) {
				const size_t ne = ne_rander(random_gen);
				array1[d] = ne;
			}
			test_shape_array.emplace_back(Shape{dim, array, NumberType::FP32}, Shape{dim, array1, NumberType::FP32});
		}
	}

	template<OperatorType T_op_type>
	void Execute() {
		for (const auto [shape_0, shape_1]: test_shape_array) {
			mat_op_test<T_op_type>(shape_0, shape_1);
		}
	}
};

TEST_F(BasicTest, MatMul) {
	Execute<OperatorType::MatMul>();
}
