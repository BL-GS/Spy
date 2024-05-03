/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#include <iostream>
#include <coroutine>

#include "async/loop.h"

using namespace spy;

int main() {

	SystemLoop loop;
	loop.start(2);

	loop.co_spawn([]() -> Task<> {
		std::cout << "First";
		std::cout << "Second";
	});

	loop.stop();

	return 0;
}