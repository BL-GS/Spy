/*
 * @author: BL-GS
 * @date:   24-5-1
 */

#include <iostream>
#include <chrono>
#include <coroutine>
#include "async/loop.h"
#include <liburing.h>

void uring_test() {
	io_uring ring;
	io_uring_queue_init(24, &ring, 0);

	io_uring_sqe *sqe_ptr = io_uring_get_sqe(&ring);

	int64_t offset = lseek(STDOUT_FILENO, 0, SEEK_CUR);
	io_uring_prep_write(sqe_ptr, STDOUT_FILENO, "Hello World!", 12, offset);

	int data = 0x1234'5678;
	io_uring_sqe_set_data(sqe_ptr, &data);

	io_uring_submit(&ring);

	io_uring_cqe *cqe_ptr = nullptr;
	io_uring_wait_cqes(&ring, &cqe_ptr, 1, nullptr, nullptr);

	unsigned head = 0;
	io_uring_for_each_cqe(&ring, head, cqe_ptr) {
		std::cout << "write data: " << cqe_ptr->res << " nbytes\n";
		int submit_data = 0;
		submit_data = *(int *)cqe_ptr->user_data;
		std::cout << "submit data: " << std::hex <<  submit_data << '\n';
	}
}

using namespace spy;

Task<> write_stdout() {
	co_await uring_write(STDOUT_FILENO, {"Hello World!\n", 13}, 0);
	co_await uring_write(STDOUT_FILENO, {"Another greeting\n", 17}, 0);
}

int main() {
	const auto loop_build_start = std::chrono::steady_clock::now();
	SystemLoop loop;
	loop.start(1);
	const auto loop_build_end = std::chrono::steady_clock::now();

	const auto loop_task_start = std::chrono::steady_clock::now();
	loop.co_synchronize(write_stdout());
	const auto loop_task_end = std::chrono::steady_clock::now();
	
	spy_info("Build time:   {}ns", std::chrono::duration_cast<std::chrono::nanoseconds>(loop_build_end - loop_build_start).count());
	spy_info("Execute time: {}ns", std::chrono::duration_cast<std::chrono::nanoseconds>(loop_task_end - loop_task_start).count());
	uring_test();
	return 0;
}