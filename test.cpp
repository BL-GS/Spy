/*
 * @author: BL-GS
 * @date:   24-5-1
 */

#include <iostream>
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
	SystemLoop loop;
	loop.start(1);
	loop.co_synchronize(write_stdout());
//	uring_test();
	return 0;
}