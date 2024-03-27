#include <cstdint>
#include <random>
#include <vector>
#include <gtest/gtest.h>

#include "operator/type.h"
#include "operator_test_suite.h"
#include "backend/cpu/cpu.h"
#include "graph/scheduler.h"

using namespace spy;

template<OperatorType T_op_type, class Binary_Op>
void basic_op_test(const Shape &shape, Binary_Op op) {
	const size_t num = shape.total_element();

	std::vector<float> operand_0(num, 0);
	std::vector<float> operand_1(num, 0);
	std::vector<float> result;
	std::vector<float> reference(num, 0);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist;

	for (int i = 0; i < num; ++i) {
		operand_0[i] = dist(gen);
		operand_1[i] = dist(gen);
		reference[i] = op(operand_0[i], operand_1[i]);
	}

	DefaultCPUBackend cpu_backend(1, 0);

	Graph graph;
	OperatorTestGraph base;

	const NodeCredit input_operand_0 = base.create_input_tensor(graph, "Operand 0", NumberType::FP32, shape.dim, shape.elements);
	graph.get_node_content<DataNode>(input_operand_0)->tensor.set_data_ptr(operand_0.data());
	const NodeCredit input_operand_1 = base.create_input_tensor(graph, "Operand 1", NumberType::FP32, shape.dim, shape.elements);
	graph.get_node_content<DataNode>(input_operand_1)->tensor.set_data_ptr(operand_1.data());
	const NodeCredit output = base.make_stream<T_op_type>(graph, "Op", { input_operand_0, input_operand_1 });
	Tensor &output_tensor = graph.get_node_content<DataNode>(output)->tensor;
	result.resize(output_tensor.total_element(), 0);
	output_tensor.set_data_ptr(result.data());

	graph.set_end(output);

	GraphScheduler scheduler(&graph, &cpu_backend);
	scheduler.execute();

	for (int i = 0; i < num; ++i) {
		EXPECT_EQ(reference[i], result[i]);
	}
}

class BasicTest: public testing::Test {
public:
	static constexpr size_t MIN_DIM         = 1;
	static constexpr size_t MAX_DIM         = 2;
	static constexpr size_t MIN_ELEMENT     = 16;
	static constexpr size_t MAX_ELEMENT     = 1024;
	static constexpr size_t NUM_TEST        = 16;

public:
	std::vector<Shape>                      test_shape_array;

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

			test_shape_array.emplace_back(dim, array, NumberType::FP32);
		}
	}

	template<OperatorType T_op_type, class Binary_Op>
	void Execute(Binary_Op op) {
		for (const auto &shape: test_shape_array) {
			basic_op_test<T_op_type>(shape, op);
		}
	}
};

TEST_F(BasicTest, Add) {
	Execute<OperatorType::Add>(std::plus<float>());
}

TEST_F(BasicTest, Sub) {
	Execute<OperatorType::Sub>(std::minus<float>());
}

TEST_F(BasicTest, Mul) {
	Execute<OperatorType::Mul>(std::multiplies<float>());
}

TEST_F(BasicTest, Div) {
	Execute<OperatorType::Div>(std::divides<float>());
}