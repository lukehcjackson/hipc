#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

__global__ void monte(*int count_d, float* randomNums) {

    //inside the kernel function
    //this is running once for each thread, so 1000 times per block, and there are 1000 blocks
    //so 1 million times overall

    int i;
    double x, y, z;

    //find the global id of this thread
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    i = tid;
    int xidx = 0, yidx = 0;

    //do the monte carlo
    xidx = i+i;
    yidx = xidx + 1;

    //get random x, y points
    x = randomNums[xidx];
    y = randomNums[yidx];
    z = (x*x) + (y*y);

    if (z <= 1) {
        count_d[tid] = 1;
    } else {
        count_d[tid] = 0;
    }

}

//Used to check if there are any errors launching the kernel
void CUDAErrorCheck()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error : %s (%d)\n", cudaGetErrorString(error), error);
        exit(0);
    }
}

int main(int argc, char* argv[]) {

    //define variables
    double pi;
    struct timespec tstart={0,0}, tend={0,0};

    int threads = 1000;
    int blocks = 1000;
    ///n_iter = threads x blocks
    long n_iter = 1000000;

    //we want to use a random x and y each iteration
    //so pre-generate an array of 2 x iters random numbers
    //allocate space for this array on the gpu:
    float* randomNums;
    cudaMalloc((void**)&randomNums, 2*n_iter*sizeof(float));

    //now use CuRand to populate that array on the device
    int status;
    curandGenerator_t gen;
    status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    status |= curandSetPseudoRandomGeneratorSeed(gen, 4294967296ULL^time(NULL));
    status |= curandGenerateUniform(gen, randomNums, (2*n_iter));
    status |= curandDestroyGenerator(gen);

    //check if there was a problem launching curand kernels and generating numbers
    if (status != CURAND_STATUS_SUCCESS) {
        printf("CURAND FAILURE!!");
        exit(EXIT_FAILURE);
    }

    //now that we have done that, we can do the actual work
    //create arrays for counting which points are inside circle
    int* count_d;
    int* count = (int*)malloc(blocks * threads * sizeof(int));
    unsigned int reducedCount = 0;

    clock_gettime(CLOCK_MONOTONIC, &tstart);

    //allocate the count_d array to hold 1 if the point is in the circle, or 0 if not
    cudaMalloc((void**)&count_d, blocks * threads * sizeof(int));

    CUDAErrorCheck();
    //launch the kernel function
    monte <<<blocks, threads>>> (count_d, randomNums);

    //device synchronise acts as a barrier until the kernel is finished
    //kernel calls are non-blocking, so the code would continue regardless of whether
    //the kernel succeeded or not without the sync
    cudaDeviceSynchronize();
    CUDAErrorCheck();

    //copy the result array back to CPU
    cudaMemcpy(count, count_d, blocks * threads * sizeof(int), cudaMemcpyDeviceToHost);

    //reduce the array into an int
    for (int i = 0; i < n_iter; i++) {
        reducedCount += count[i];
    }

    //free cuda malloc'd arrays
    cudaFree(randomNums);
    cudaFree(count_d);
    free(count);

    //find pi
    pi = ((double)reducedCount / n_iter) * 4.0;

    clock_gettime(CLOCK_MONOTONIC, &tend);

    printf("Pi: %f\n", pi);
    printf("Time taken %.6f ms\n",
           ((double)tend.tv_sec * 0.001 + 1.0e-6*tend.tv_nsec) -
           ((double)tstart.tv_sec * 0.001 + 1.0e-6*tstart.tv_nsec));

    return 0;
}

