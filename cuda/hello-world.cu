#include <stdio.h>

__global__ void cuda_hello() {

    int thread_id = threadIdx.x;
    int stride = blockDim.x;
    int block_id = blockIdx.x;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello, World! From GPU! block id: %d thread id: %d\n", block_id, thread_id);
}

int main(int argc, char *argv[]) {
    cuda_hello<<<3,3>>>();
    cudaDeviceSynchronize();
    return 0;
}
