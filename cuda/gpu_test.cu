#include <cstdio>
#include <cuda_runtime.h>
#include "gpu_test.cuh"

__global__ void test_kernel() {
	printf("Hello from GPU!\n");
}

void gpu_test() {
	test_kernel<<<1, 1>>>();
	cudaDeviceSynchronize();
}

